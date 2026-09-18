"""Render saved predictions on fsaverage5 with matched color scales."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(ROOT / 'cache/matplotlib')
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image
from nilearn import datasets, plotting


def main():
    rows = json.loads((ROOT / 'inputs/manifest.json').read_text())
    arrays = [np.load(ROOT / f"outputs/prediction_{r['index']}.npz") for r in rows]
    preds = [a['predictions'] for a in arrays]
    assert preds[0].shape == preds[1].shape
    assert all(np.isfinite(p).all() for p in preds)
    maps = [p.mean(axis=0) for p in preds]
    delta = maps[0]-maps[1]
    metrics = {
        'spatial_pearson_r': float(np.corrcoef(*maps)[0, 1]),
        'mean_absolute_difference': float(np.abs(delta).mean()),
        'root_mean_square_difference': float(np.sqrt(np.mean(delta**2))),
        'mean_signed_difference_751_minus_4849': float(delta.mean()),
        'per_image': {str(r['index']): {'shape': list(p.shape),
                     'mean': float(p.mean()), 'rms_temporal_mean_map': float(np.sqrt(np.mean(m*m))),
                     'human_complexity': r['mean_human_complexity_rating']}
                     for r, p, m in zip(rows, preds, maps)},
    }
    (ROOT/'outputs/comparison.json').write_text(json.dumps(metrics, indent=2)+'\n')
    np.savez_compressed(ROOT/'outputs/mean_maps.npz', image_751=maps[0], image_4849=maps[1], difference=delta)
    fs = datasets.fetch_surf_fsaverage(mesh='fsaverage5', data_dir=str(ROOT/'cache/nilearn'))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    fig = plt.figure(figsize=(16, 16), facecolor='#fafafa')
    gs = fig.add_gridspec(5, 5, width_ratios=[1,1,1,1,.13], height_ratios=[1.4, 1, 1, 1, .8], hspace=.22, wspace=.03)
    fig.suptitle('MASSVIS · Predicted cortical responses', fontsize=23, x=.07, ha='left', y=.98)
    for i, r in enumerate(rows):
        ax = fig.add_subplot(gs[0, i*2:(i+1)*2])
        ax.imshow(Image.open(ROOT/f"inputs/{r['index']}.png")); ax.axis('off')
        ax.set_title(f"Image {r['index']} · human complexity {r['mean_human_complexity_rating']:.1f}/100", pad=10)
    limit = max(float(np.abs(m).max()) for m in maps)
    difference_limit = float(np.abs(delta).max())
    views = [('left', 'lateral'), ('left', 'medial'), ('right', 'medial'), ('right', 'lateral')]
    for row, (values, title, lim) in enumerate(zip(maps+[delta], ['751 · temporal mean', '4849 · temporal mean', 'Difference · 751 − 4849'], [limit, limit, difference_limit]), start=1):
        axes=[]
        for col, (hemi, view) in enumerate(views):
            ax = fig.add_subplot(gs[row, col], projection='3d'); axes.append(ax)
            half = values[:10242] if hemi == 'left' else values[10242:]
            plotting.plot_surf_stat_map(fs[f'infl_{hemi}'], half, hemi=hemi, view=view,
                bg_map=fs[f'sulc_{hemi}'], cmap='RdBu_r', vmin=-lim, vmax=lim,
                symmetric_cbar=True, colorbar=False, axes=ax, figure=fig, threshold=None)
            ax.set_title(f'{hemi.capitalize()} {view}', fontsize=10, pad=0)
        axes[0].text2D(-.05, 1.13, title, transform=axes[0].transAxes, fontsize=13, weight='bold')
        host = fig.add_subplot(gs[row,4]); host.axis('off')
        cax = host.inset_axes([.05,.18,.22,.64])
        fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-lim,lim), cmap='RdBu_r'),
                     cax=cax, label='Model output units')
    ax = fig.add_subplot(gs[4, :2])
    for r, p, a in zip(rows, preds, arrays):
        ax.plot(a['times'], np.sqrt(np.mean(p*p, axis=1)), label=str(r['index']), lw=2)
    ax.set(xlabel='Stimulus-aligned time (s)', ylabel='Cortical RMS (model units)')
    ax.legend(frameon=False); ax.spines[['top','right']].set_visible(False)
    ax = fig.add_subplot(gs[4, 2:4]); ax.axis('off')
    ax.text(.08,.9, f"Spatial correlation: {metrics['spatial_pearson_r']:.3f}\nMean absolute difference: {metrics['mean_absolute_difference']:.4f}\nRMS difference: {metrics['root_mean_square_difference']:.4f}", va='top', fontsize=13, linespacing=1.6)
    fig.text(.07,.035, 'Pretrained TRIBE v2 · average subject · fsaverage5 (20,484 vertices) · identical 12 s silent static presentations\nMean over all 12 predictions; separate timelines. Native model timing includes its 5 s hemodynamic compensation.\nExploratory model predictions, not measured fMRI or validated complexity scores. Difference uses its own symmetric scale.', fontsize=10, linespacing=1.5)
    fig.subplots_adjust(top=.935, bottom=.105, left=.055, right=.92)
    fig.savefig(ROOT/'outputs/comparison.png', dpi=160, facecolor=fig.get_facecolor())
    fig.savefig(ROOT/'outputs/comparison.pdf', facecolor=fig.get_facecolor())
    plt.close(fig)
    (ROOT/'index.html').write_text(f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TRIBE v2 · MASSVIS comparison</title>
<style>body{{margin:40px auto;padding:0 24px;max-width:1280px;font:16px/1.6 system-ui;color:#18262b;background:#fafafa}}h1{{line-height:1.15}}img{{width:100%;height:auto}}a{{color:#006980}}.metrics{{display:flex;gap:36px;flex-wrap:wrap}}.metrics strong{{font-size:28px;display:block}}p{{max-width:850px}}</style>
<h1>TRIBE v2 · MASSVIS 751 &amp; 4849</h1>
<p>Pretrained average-subject cortical predictions for identical 12-second silent
presentations. Human complexity ratings are 92.5 and 1.9 / 100, respectively.</p>
<div class="metrics"><div><strong>{metrics['spatial_pearson_r']:.3f}</strong>Spatial correlation</div>
<div><strong>{metrics['mean_absolute_difference']:.4f}</strong>Mean absolute difference</div>
<div><strong>{metrics['root_mean_square_difference']:.4f}</strong>RMS difference</div></div>
<p>Differences are in model output units. These predictions are neither measured
fMRI nor validated complexity scores. The two response maps share a color scale;
the signed difference uses a separate scale.</p>
<img src="outputs/comparison.png" alt="Both source images, cortical predictions on left and right hemisphere surfaces, their signed difference, and response magnitude over time">
<p><a href="outputs/comparison.png">Full-resolution image</a> ·
<a href="outputs/comparison.pdf">PDF</a> · <a href="outputs/comparison.json">Metrics JSON</a> ·
<a href="README.md">Protocol and reproduction</a></p></html>''')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
