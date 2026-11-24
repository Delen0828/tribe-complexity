"""Nested grouped out-of-fold explanations of full-resolution MassVis Ridge."""
import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

from compare_raw_tribe import ALPHAS, load_responses
from compare_representations import SEED, inner_folds
from complexity_features import TRIBE, WORKSPACE, cortical_regions, prediction_records, source_rows
from replicate import HERE, load_config, load_dataset, sha256

REPEATS = 30
OUT = WORKSPACE / "ridge_interpretability"


def save_npz(path, **data):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        np.savez(tmp, **data)
    Path(tmp.name).replace(path)


def removed_weights(projected, inverse, full):
    """Woodbury correction in ROI feature space, solving all alphas together.

    In the full Gram eigenbasis, B = Q.T @ X_roi and D = (eigenvalues + alpha)^-1.
    Removing the ROI gives dual weights full + D B (I - B.T D B)^-1 B.T full.
    Apply the positive definite system with conjugate gradients to avoid forming
    a separate dense ROI matrix for each regularization setting.
    """
    rhs = projected.T @ full
    residual = rhs.clone()
    direction = residual.clone()
    solution = torch.zeros_like(rhs)
    squared = residual.square().sum(0)
    tolerance = squared * 1e-20
    for _ in range(256):
        active = squared > tolerance
        if not active.any():
            break
        applied = direction - projected.T @ (inverse * (projected @ direction))
        step = torch.where(active, squared / (direction * applied).sum(0), 0)
        solution += direction * step
        residual -= applied * step
        updated = residual.square().sum(0)
        direction = residual + direction * torch.where(active, updated / squared, 0)
        squared = updated
    else:
        raise ArithmeticError("ROI Ridge solve did not converge")
    return full + inverse * (projected @ solution), solution


def fit_split(responses, y, fit, test, labels, device, selected=None):
    """Return fold MAEs for tuning, or paired predictions and full-model patterns."""
    alphas = torch.tensor(ALPHAS, device=device, dtype=torch.float64)
    target = torch.tensor(y[fit], device=device, dtype=torch.float64)
    center_y = target.mean()
    n = len(fit)
    gram = torch.zeros((n, n), dtype=torch.float64, device=device)
    cross = gram.new_zeros((len(test), n))
    regions = range(1, int(labels.max()) + 1)

    def features(vertices):
        values = torch.as_tensor(responses[:, :, vertices].transpose(0, 2, 1).copy(),
                                 device=device, dtype=torch.float64).flatten(1)
        train = values[fit]
        variance, mean = torch.var_mean(train, dim=0, correction=0)
        scale = torch.where(variance > 0, variance.sqrt(), 1)
        return (train - mean) / scale, (values[test] - mean) / scale, scale

    for vertices in np.array_split(np.arange(len(labels)), max(1, len(labels) // 512)):
        train, evaluate, _ = features(vertices)
        gram.addmm_(train, train.T)
        cross.addmm_(evaluate, train.T)
    # Standardized training columns are centered, so the intercept is mean(y).
    eigen, basis = eigh(gram.cpu().numpy(), driver="evd")
    basis = torch.as_tensor(basis, device=device)
    eigen = torch.as_tensor(eigen.clip(0), device=device)
    inverse = (eigen[:, None] + alphas).reciprocal()
    full = inverse * (basis.T @ (target - center_y))[:, None]
    evaluate_basis = cross @ basis
    full_pred = evaluate_basis @ full + center_y
    observed = torch.as_tensor(y[test], device=device)[:, None]
    if selected is None:
        errors = [(observed - full_pred).abs().mean(0).cpu().numpy()]
    else:
        i = selected[0]
        prediction = full_pred[:, i].cpu().numpy()
        train_pred = basis @ (eigen * full[:, i])
        train_pred -= train_pred.mean()
        variance_sum = train_pred.square().sum()
        if variance_sum <= 0:
            raise ValueError("Haufe pattern undefined for constant predictions")
        dual = basis @ full[:, i]
        pattern = np.empty(responses.shape[1:], dtype=np.float64)
        parts = np.empty((len(test), len(regions)))
        removed = np.empty_like(parts)

    for region in range(len(regions) + 1):
        vertices = np.flatnonzero(labels == region)
        if not len(vertices) or (region == 0 and selected is None):
            continue
        train, evaluate, scale = features(vertices)
        if selected is not None:
            raw_pattern = (train.T @ train_pred) * scale / variance_sum
            pattern[:, vertices] = raw_pattern.reshape(len(vertices), -1).T.cpu().numpy()
            if region == 0:
                continue
            parts[:, region - 1] = (evaluate @ (train.T @ dual)).cpu().numpy()
        projected = basis.T @ train
        columns = slice(None) if selected is None else [selected[region]]
        weights, correction = removed_weights(projected, inverse[:, columns], full[:, columns])
        # X_roi.T @ reduced_dual equals the Woodbury correction.
        pred = evaluate_basis @ weights - evaluate @ correction + center_y
        if selected is None:
            errors.append((observed - pred).abs().mean(0).cpu().numpy())
        else:
            removed[:, region - 1] = pred[:, 0].cpu().numpy()
    if selected is None:
        return {"errors": np.stack(errors)}
    return dict(full=prediction, removed=removed, parts=parts, pattern=pattern,
                alpha=np.asarray(ALPHAS)[selected], test=test)


def permutation_predictions(full, parts, order):
    # Replacing this additive Ridge term permutes the entire ROI as one block.
    return full[:, None] + parts[order] - parts


def summarize(folds, y, groups, ids, names):
    full = np.empty_like(y)
    removed = np.empty((len(y), len(names) - 1))
    perm_sse = np.zeros((REPEATS, removed.shape[1]))
    patterns, sizes, coverage = [], [], np.zeros(len(y), dtype=int)
    fold_ids = np.empty(len(y), dtype=int)
    for fold, result in enumerate(folds):
        test = result["test"]
        coverage[test] += 1
        fold_ids[test] = fold
        pred = result["full"]
        full[test], removed[test] = pred, result["removed"]
        rng = np.random.default_rng([SEED, fold])
        for repeat in range(REPEATS):
            shuffled = permutation_predictions(pred, result["parts"], rng.permutation(len(test)))
            perm_sse[repeat] += ((y[test, None] - shuffled) ** 2).sum(0)
        patterns.append(result["pattern"])
        sizes.append(len(y) - len(test))
    if not np.all(coverage == 1) or any(len(set(fold_ids[groups == g])) != 1 for g in np.unique(groups)):
        raise ValueError("Invalid grouped outer-fold coverage")
    sst = ((y - y.mean()) ** 2).sum()
    sse = ((y - full) ** 2).sum()
    ablation = (((y[:, None] - removed) ** 2).sum(0) - sse) / sst
    permutation = (perm_sse.mean(0) - sse) / sst
    pattern = np.average(patterns, axis=0, weights=sizes)
    report = dict(images=len(y), groups=len(np.unique(groups)), folds=len(folds),
                  repeats=REPEATS, r2=1 - sse / sst,
                  limits=dict(ablation=abs(ablation).max(), permutation=abs(permutation).max(),
                              haufe=abs(pattern).max()),
                  regions=[dict(name=names[r], ablation=ablation[r-1], permutation=permutation[r-1])
                           for r in range(1, len(names))])
    audit = dict(image_id=np.asarray(ids, dtype=str), source_sha256=groups, observed=y, fold=fold_ids,
                 full=full, removed=removed, region_names=list(names.values())[1:], pattern=pattern,
                 permutation_sse=perm_sse)
    return report, audit


def run(device):
    atlas = TRIBE / "cache/nilearn"
    hemis, names = cortical_regions(atlas)
    labels = np.concatenate([hemis["left"], hemis["right"]])
    config, base = load_config(HERE / "config.json")
    frame, _, ratings, inputs = load_dataset(config, base)
    print("Validating MassVis sources and TRIBE prediction receipts", flush=True)
    rows = source_rows(frame, WORKSPACE)
    records, _ = prediction_records(rows, WORKSPACE)
    if missing := set(frame.index) - records.keys():
        raise ValueError(f"Missing TRIBE predictions for {sorted(missing)[:10]}")
    groups = np.asarray([records[i].source_hash for i in frame.index])
    y = ratings.to_numpy()
    provenance = dict(seed=SEED, alphas=ALPHAS, repeats=REPEATS,
                      inputs={k: sha256(p) for k, p in inputs.items()},
                      atlas={p.name: sha256(p) for p in sorted((atlas / "destrieux_surface").glob("*.annot"))},
                      code={p.name: sha256(p) for p in [Path(__file__), HERE / "complexity_features.py",
                            HERE / "compare_raw_tribe.py", HERE / "compare_representations.py"]},
                      prediction_sha256=[records[i].prediction_hash for i in frame.index],
                      source_sha256=groups.tolist())
    signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    folder = OUT / signature
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "provenance.json").write_text(json.dumps(provenance))
    responses = load_responses(frame.index, rows, records)
    results = []
    for fold, (train, test) in enumerate(inner_folds(frame.index.to_numpy(), groups)):
        path = folder / f"outer-{fold}.npz"
        if not path.exists():
            errors = []
            for inner, (fit, val) in enumerate(inner_folds(train, groups)):
                checkpoint = folder / f"inner-{fold}-{inner}.npz"
                if not checkpoint.exists():
                    print(f"Outer {fold+1}/5 · inner {inner+1}/5", flush=True)
                    output = fit_split(responses, y, train[fit], train[val], labels, device)
                    save_npz(checkpoint, **output, validation=train[val])
                with np.load(checkpoint) as saved:
                    errors.append(saved["errors"])
            selected = np.mean(errors, axis=0).argmin(axis=1)
            print(f"Outer {fold+1}/5 · refitting", flush=True)
            save_npz(path, **fit_split(responses, y, train, test, labels, device, selected))
        with np.load(path) as saved:
            results.append(dict(saved))
    report, audit = summarize(results, y, groups, frame.image_id, names)
    save_npz(folder / "oof.npz", **audit, vertex_labels=labels)
    from render_interpretability import publish
    publish(folder, report, audit["pattern"], labels)
    print(f"Published {report['images']} OOF predictions, R² = {report['r2']:.6f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        run(args.device)
