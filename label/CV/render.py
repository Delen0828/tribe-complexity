"""Candidate contact sheets, auditable exports, and a buildless static data bundle."""
from __future__ import annotations

import concurrent.futures
import csv
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from common import CV, read_json, write_json, digest, decode_mask, rle_bbox, now, FAMILIES


def font(size=14):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def fit(image, size):
    out = image.copy()
    out.thumbnail(size, Image.Resampling.LANCZOS)
    return out


def color(instance_id):
    import colorsys
    hue = int(digest(instance_id)[:6], 16) / 0xFFFFFF
    return tuple(int(v*255) for v in colorsys.hsv_to_rgb(hue, .8, .9))


def sheets(args):
    from run import active_run
    directory = active_run(args.run_id)
    rows = read_json(CV / "selection.json")["records"]
    jobs = []
    for row in rows:
        if args.ids and row["image_id"] not in args.ids.split(","):
            continue
        path = directory / "candidates" / f"{row['image_id']}.json"
        if not path.exists():
            continue
        raw = read_json(path)
        candidates = [c for c in raw["candidates"] if not c.get("duplicate_of") and not c.get("automatic_rejection")]
        target = directory / "triage/sheets" / row["image_id"]
        target.mkdir(parents=True, exist_ok=True)
        job = {"figure_id": row["image_id"], "candidate_set_id": raw["candidate_set_id"], "reference_path": str(CV / "results/images" / row["image_id"]), "preparation_path": str(CV / "preparation/figures" / f"{row['image_id']}.json"), "output_path": str(directory / "triage" / f"{row['image_id']}.json"), "sheets": [], "candidates": []}
        image = Image.open(job["reference_path"]).convert("RGB")
        for offset in range(0, len(candidates), 16):
            subset = candidates[offset:offset+16]
            sheet_path = target / f"sheet_{offset//16:03d}.jpg"
            sheet = Image.new("RGB", (1200, 1040), "#f1f3f6")
            draw = ImageDraw.Draw(sheet)
            draw.text((12, 10), f"Figure {row['image_id']} | native {row['width']} x {row['height']} | candidates {offset+1}-{offset+len(subset)}", font=font(18), fill="#17252a")
            draw.text((12, 36), "Each tile: local context (left), isolated candidate mask (right). Retain SINGLE discrete data marks only.", font=font(14), fill="#37474f")
            for i, c in enumerate(subset):
                col, line = i % 4, i // 4
                left, top = col*300+8, line*240+68
                b = rle_bbox(c["rle"])
                pad = max(12, min(40, int(max(b[2]-b[0], b[3]-b[1])*.25)))
                context = image.crop((max(0,b[0]-pad), max(0,b[1]-pad), min(image.width,b[2]+pad), min(image.height,b[3]+pad)))
                contextual = fit(context, (136, 165))
                draw.rounded_rectangle((left,top,left+284,top+228), 6, fill="white", outline="#cad2d9")
                draw.text((left+5,top+4), c["candidate_id"], font=font(15), fill="#102027")
                draw.text((left+5,top+25), f"area {c['area']} | quality {c['score']:.2f}", font=font(12), fill="#546e7a")
                sheet.paste(contextual, (left+5+(136-contextual.width)//2,top+47+(165-contextual.height)//2))
                tight = image.crop(b).convert("RGBA")
                mask = Image.fromarray(decode_mask(c["rle"]).astype(np.uint8)*255)
                tight.putalpha(mask)
                isolated = fit(tight, (136,165))
                bg = Image.new("RGB", isolated.size, "#dee5e8")
                bg.paste(isolated, mask=isolated.getchannel("A"))
                sheet.paste(bg, (left+145+(136-isolated.width)//2,top+47+(165-isolated.height)//2))
                draw.text((left+5,top+211), f"box {','.join(map(str,b))}"[:42], font=font(10), fill="#546e7a")
                job["candidates"].append({"candidate_id": c["candidate_id"], "panel_id": c["panel_id"], "bbox": b, "area": c["area"], "score": c["score"], "families": c["families"], "origin": c["origin"], "sheet": str(sheet_path)})
            sheet.save(sheet_path, quality=92)
            job["sheets"].append(str(sheet_path))
        write_json(target / "job.json", job)
        if candidates:
            jobs.append(job)
        else:
            write_json(directory / "triage" / f"{row['image_id']}.json", {"figure_id": row["image_id"], "candidate_set_id": raw["candidate_set_id"], "reviewed_by": "no_unique_candidates", "decisions": {}})
        print(f"{row['image_id']}: {len(candidates)} unique candidates, {len(job['sheets'])} sheets", flush=True)
    suffix = "." + digest(args.ids)[:10] if args.ids else ""
    write_json(directory / f"triage/jobs{suffix}.json", jobs)


def read_decisions(directory, row, raw):
    path = directory / "triage" / f"{row['image_id']}.json"
    if not path.exists():
        return {}, None
    triage = read_json(path)
    if triage["candidate_set_id"] != raw["candidate_set_id"]:
        raise ValueError(f"Stale triage for {row['image_id']}")
    decisions = triage["decisions"]
    candidate_ids = {c["candidate_id"] for c in raw["candidates"]}
    for cid, decision in decisions.items():
        if cid not in candidate_ids or decision["decision"] not in ("retain", "reject", "ambiguous") or not decision.get("reason"):
            raise ValueError(f"Invalid triage decision {cid}")
        if decision["decision"] == "retain" and decision.get("family") not in FAMILIES:
            raise ValueError(f"Retained candidate {cid} needs a valid mark family")
    return decisions, triage


def render_figure(row, directory):
    raw_path = directory / "candidates" / f"{row['image_id']}.json"
    image = Image.open(CV / "results/images" / row["image_id"]).convert("RGB")
    result = {"figure_id": row["image_id"], "width": row["width"], "height": row["height"], "count": None, "status": "pending", "panels": [], "instances": [], "runtime_seconds": 0, "assets": {"original": f"results/images/{row['image_id']}"}, "notes": "Awaiting preparation or segmentation"}
    if raw_path.exists():
        raw = read_json(raw_path)
        decisions, triage = read_decisions(directory, row, raw)
        instances = []
        for candidate in raw["candidates"]:
            if candidate.get("duplicate_of"):
                continue
            c = dict(candidate)
            if c.get("duplicate_of"):
                c.update(decision="reject", reason="near-identical mask", triage_source="mask_iou_deduplication")
            elif c.get("automatic_rejection"):
                c.update(decision="reject", reason=c["automatic_rejection"], triage_source="prepared_exclusion_region")
            elif c["candidate_id"] in decisions:
                c.update(decisions[c["candidate_id"]], triage_source=triage.get("reviewed_by", "gpt-6-luna"))
            else:
                c.update(decision="pending", reason="Awaiting model-assisted visual triage")
            c["bbox"] = rle_bbox(c["rle"])
            c["color"] = color(c["candidate_id"])
            instances.append(c)
        counts = Counter(c["decision"] for c in instances)
        status = raw["status"]
        if status == "segmented":
            status = "awaiting_triage" if counts["pending"] else "partial" if counts["ambiguous"] else "complete"
        unresolved_panels = sum(p.get("preparation_status") == "unresolved" for p in raw["panels"])
        if unresolved_panels and status in ("complete", "not_applicable"):
            status = "partial"
        quality_issues = triage.get("quality_issues", []) if triage else []
        if quality_issues and status == "complete":
            status = "partial"
        for panel in raw["panels"]:
            own = [c for c in instances if c["panel_id"] == panel["panel_id"]]
            panel["count"] = sum(c["decision"] == "retain" for c in own) if panel["status"] == "segmented" and not any(c["decision"] == "pending" for c in own) else None
            panel["ambiguous_count"] = sum(c["decision"] == "ambiguous" for c in own)
            panel["pending_count"] = sum(c["decision"] == "pending" for c in own)
        countable = any(p["status"] == "segmented" for p in raw["panels"]) and not counts["pending"]
        result.update(status=status, count=counts["retain"] if countable else None, known_retained_count=counts["retain"], ambiguous_count=counts["ambiguous"], pending_count=counts["pending"], rejected_count=counts["reject"], unresolved_panels=unresolved_panels, panels=raw["panels"], instances=instances, runtime_seconds=raw["runtime_seconds"], candidate_set_id=raw["candidate_set_id"], preparation_sha256=raw["preparation_sha256"], errors=raw["errors"], triage_sha256=digest(triage) if triage else None, notes="Model-assisted pilot decisions; no human ground truth. Partial counts exclude unresolved masks or panel sections.")
        result["result_id"] = digest({"candidate_set_id": raw["candidate_set_id"], "triage": triage})
        result["raw_candidate_count"] = len(raw["candidates"])
        result["duplicate_count"] = sum(bool(c.get("duplicate_of")) for c in raw["candidates"])
        result["raw_candidates_json"] = str(raw_path.relative_to(CV))
        result["quality_issues"] = quality_issues
        result["triage_notes"] = triage.get("notes", "") if triage else ""
    overlay = image.convert("RGBA")
    union = np.zeros((image.height, image.width), dtype=bool)
    for c in result["instances"]:
        if c["decision"] != "retain":
            continue
        x0, y0, x1, y1 = c["bbox"]
        mask = decode_mask(c["rle"])
        union[y0:y1, x0:x1] |= mask
        local = Image.new("RGBA", (x1-x0,y1-y0), tuple(c["color"])+(0,))
        local.putalpha(Image.fromarray(mask.astype(np.uint8)*150))
        overlay.alpha_composite(local, (x0,y0))
    segmented = image.convert("RGBA")
    segmented.putalpha(Image.fromarray(union.astype(np.uint8)*255))
    asset_dir = directory / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    for key, img in [("overlay",overlay), ("segmented",segmented), ("thumbnail",fit(image,(230,160)))]:
        path = asset_dir / f"{row['image_id']}.{key}.png"
        img.save(path)
        result["assets"][key] = str(path.relative_to(CV))
    target = directory / "figures" / f"{row['image_id']}.json"
    write_json(target, result)
    js_path = target.with_suffix(".js")
    js_path.write_text("window.CV_FIGURES=window.CV_FIGURES||{};window.CV_FIGURES["+json.dumps(row["image_id"])+"]="+json.dumps(result,separators=(",",":"))+";\n")
    return result


def render(args):
    from run import active_run
    directory = active_run(args.run_id)
    selection = read_json(CV / "selection.json")
    rows = selection["records"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda row: render_figure(row,directory), rows))
    entries = []
    for row, result in zip(rows, results):
        entries.append({k:row[k] for k in ["image_id","filename","category","mean_human_complexity_rating","band","width","height"]} | {k:result.get(k,0) for k in ["count","known_retained_count","ambiguous_count","pending_count","rejected_count","status","runtime_seconds","assets","result_id"]} | {"families": sorted({family for p in result["panels"] for family in p.get("mark_families",[])}), "detail_script": str((directory/"figures"/f"{row['image_id']}.js").relative_to(CV)), "detail_json": str((directory/"figures"/f"{row['image_id']}.json").relative_to(CV))})
    statuses = Counter(r["status"] for r in results)
    bands = []
    for band in range(10):
        members = [e for e in entries if e["band"]==band]
        counted = [e["count"] for e in members if e["count"] is not None]
        bands.append({"band":band,"selected":len(members),"counted":len(counted),"total_retained":sum(counted),"mean_count":sum(counted)/len(counted) if counted else None,"ambiguous":sum(e["ambiguous_count"] for e in members)})
    summary = {"selected":len(rows),"statuses":dict(statuses),"total_retained":sum(e["known_retained_count"] for e in entries),"ambiguous":sum(e["ambiguous_count"] for e in entries),"pending_candidates":sum(e["pending_count"] for e in entries),"runtime_seconds":sum(e["runtime_seconds"] for e in entries),"bands":bands}
    family_counts = Counter(c["family"] for r in results for c in r["instances"] if c["decision"] == "retain")
    summary["retained_by_family"] = dict(sorted(family_counts.items()))
    summary["unresolved_preparation_panels"] = sum(r.get("unresolved_panels",0) for r in results)
    corrections_path = CV/"results/preparation-corrections.json"
    summary["preparation_correction_events"] = len(read_json(corrections_path)["corrections"]) if corrections_path.exists() else 0
    analysis_run_id = digest({"run_id":directory.name,"results":[r.get("result_id") for r in results]})
    data = {"schema_version":1,"run_id":analysis_run_id,"base_run_id":directory.name,"selection_id":selection["selection_id"],"model":read_json(directory/"run.json")["model"],"generated_at":now(),"summary":summary,"entries":entries,"provenance":read_json(directory/"run.json").get("provenance",{}),"count_definition":"Retained single discrete graphical data marks; continuous graphics, text, legends, axes, decoration and photographic imagery excluded. Ambiguous masks excluded from counts."}
    write_json(directory/"summary.json",summary)
    write_json(directory/"manifest.json",data)
    write_json(CV/"results/manifest.json",data)
    (CV/"data.js").write_text("window.CV_DATA="+json.dumps(data,separators=(",",":"))+";\n")
    with (CV/"results/counts.csv").open("w",newline="") as f:
        fields=["image_id","filename","band","mean_human_complexity_rating","count","ambiguous_count","pending_count","status","runtime_seconds"]
        writer=csv.DictWriter(f,fields,extrasaction="ignore");writer.writeheader();writer.writerows(entries)
    write_json(CV/"results/counts.json", [{k:e[k] for k in ["image_id","band","mean_human_complexity_rating","count","ambiguous_count","pending_count","status"]} for e in entries])
    report=["# Segmentation pilot results","",f"Model: {data['model']}. Selection: {selection['selection_id'][:12]}.","",f"150 training figures, 15 per observed-complexity band. {dict(statuses)}.","",f"Retained discrete masks: {summary['total_retained']}; ambiguous: {summary['ambiguous']}; awaiting triage: {summary['pending_candidates']}.","",f"Inference time (sum of figure times): {summary['runtime_seconds']/60:.1f} minutes.","","Counts are model-assisted detections, not validated human-visible counts. Partial results exclude ambiguous masks. N/A figures remain in the sample. No accuracy metrics, K, or E are estimated.","","| Complexity | Selected | Counted figures | Retained marks | Mean count | Ambiguous |","|---|---:|---:|---:|---:|---:|"]
    for b in bands:
        mean=f"{b['mean_count']:.2f}" if b["mean_count"] is not None else "N/A"
        report.append(f"| {b['band']*10}–{(b['band']+1)*10} | {b['selected']} | {b['counted']} | {b['total_retained']} | {mean} | {b['ambiguous']} |")
    report += ["", "## Retained mark families", "", "These are graphical units, not unique underlying observations. Units differ across families.", "", "| Family | Retained masks |", "|---|---:|"]
    report += [f"| {family} | {n} |" for family,n in sorted(family_counts.items())]
    report += ["",f"Unresolved preparation panels: {summary['unresolved_preparation_panels']}. Preparation correction events: {summary['preparation_correction_events']}.","", "Panel/exemplar preparation was visually revised when candidate inspection exposed source-coordinate or chart-type errors. Older raw receipts and the correction audit are preserved. Mask confidence/stability defaults were not tuned to reference counts. Known omitted marks and unresolved instance boundaries are flagged for human post-hoc assessment."]
    (CV/"results/summary.md").write_text("\n".join(report)+"\n")
    print(f"Rendered {len(rows)} figures: {dict(statuses)}; retained {summary['total_retained']}",flush=True)
