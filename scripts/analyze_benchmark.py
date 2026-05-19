#!/usr/bin/env python3
"""Post-process a benchmark `runs.csv` into a human-readable report.

Re-runs the classifier with looser heuristics (an exit-code 0 with
``total_src=0`` is FlashVSR's "too few anchors for 8n+1" case, not a
generic error), then emits Markdown with:

- run counts by status
- top-N by FPS overall and per model
- VRAM-efficient frontier
- per-knob sensitivity (mean FPS by pre-resize / frame-skip / dtype / ...)
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def _f(v):
    try:
        return float(v)
    except Exception:
        return 0.0


def _i(v):
    try:
        return int(v)
    except Exception:
        return 0


def reclassify(rows):
    for r in rows:
        if r["status"] != "error":
            continue
        err = (r.get("error") or "").lower()
        if "out of memory" in err or "oom" in err:
            r["status"] = "oom"
        elif "total_src=0" in err or "8n+1" in err or "non-empty list" in err:
            r["status"] = "too_few_frames"
    return rows


def fmt_row(r):
    return (f"  {_f(r['e2e_fps']):6.2f} fps  vram={_i(r['peak_vram_mb']):>5} MB  "
            f"{r['model']:<15}  scale={r['scale']} pr={r['pre_resize']:<5} "
            f"sk={r['frame_skip']} sage={r['sage_attn']:<5} quant={r['quant']:<8} "
            f"dt={r['dtype']:<4}  out_res={r['out_res']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="output markdown (default: <csv-dir>/REPORT.md)")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    rows = reclassify(rows)
    out_md = args.out or (args.csv.parent / "REPORT.md")

    counts = defaultdict(int)
    for r in rows:
        counts[r["status"]] += 1

    ok = [r for r in rows if r["status"] == "ok"]
    ok.sort(key=lambda r: -_f(r["e2e_fps"]))

    lines: list[str] = []
    lines.append(f"# Benchmark report — {args.csv.parent.name}\n")
    lines.append(f"- runs: **{len(rows)}**")
    for k in ("ok", "oom", "too_few_frames", "error", "timeout"):
        if counts[k]:
            lines.append(f"- {k}: {counts[k]}")
    lines.append("")

    # Top 20 by FPS overall
    lines.append("## Top 20 by end-to-end FPS\n")
    lines.append("| FPS | VRAM MB | model | scale | pr | skip | sage | quant | dtype | out_res |")
    lines.append("|---:|---:|---|---:|---|---:|---|---|---|---|")
    for r in ok[:20]:
        lines.append(
            f"| **{_f(r['e2e_fps']):.2f}** | {_i(r['peak_vram_mb'])} | "
            f"{r['model']} | {r['scale']} | {r['pre_resize']} | {r['frame_skip']} | "
            f"{r['sage_attn']} | {r['quant']} | {r['dtype']} | {r['out_res']} |"
        )
    lines.append("")

    # Best per model
    lines.append("## Best per model\n")
    by_model = defaultdict(list)
    for r in ok:
        by_model[r["model"]].append(r)
    for m, rs in by_model.items():
        rs.sort(key=lambda r: -_f(r["e2e_fps"]))
        lines.append(f"### {m}\n")
        lines.append("| FPS | VRAM MB | scale | pr | skip | sage | quant | dt | out_res |")
        lines.append("|---:|---:|---:|---|---:|---|---|---|---|")
        for r in rs[:10]:
            lines.append(
                f"| **{_f(r['e2e_fps']):.2f}** | {_i(r['peak_vram_mb'])} | "
                f"{r['scale']} | {r['pre_resize']} | {r['frame_skip']} | "
                f"{r['sage_attn']} | {r['quant']} | {r['dtype']} | {r['out_res']} |"
            )
        lines.append("")

    # VRAM-efficient frontier — for each unique VRAM bucket, the highest FPS
    lines.append("## VRAM-efficient frontier (highest FPS at each VRAM band)\n")
    buckets: dict[int, dict] = {}
    for r in ok:
        v = _i(r["peak_vram_mb"])
        band = v // 256 * 256
        cur = buckets.get(band)
        if cur is None or _f(r["e2e_fps"]) > _f(cur["e2e_fps"]):
            buckets[band] = r
    lines.append("| ≤ VRAM MB | best FPS | model | pr | sk | quant | dt |")
    lines.append("|---:|---:|---|---|---:|---|---|")
    for band in sorted(buckets):
        r = buckets[band]
        lines.append(
            f"| {band+256} | {_f(r['e2e_fps']):.2f} | {r['model']} | "
            f"{r['pre_resize']} | {r['frame_skip']} | {r['quant']} | {r['dtype']} |"
        )
    lines.append("")

    # OFAT sensitivity (avg FPS per knob value, within each model)
    lines.append("## Per-knob sensitivity (mean FPS over `ok` runs)\n")
    knobs = ["pre_resize", "frame_skip", "sage_attn", "quant", "dtype",
             "chunk_frames", "topk_ratio", "kv_ratio", "local_range",
             "color_fix", "crf", "scale"]
    for m, rs in by_model.items():
        lines.append(f"### {m}\n")
        for k in knobs:
            agg = defaultdict(list)
            for r in rs:
                agg[r[k]].append(_f(r["e2e_fps"]))
            if len(agg) < 2:
                continue
            lines.append(f"**{k}**")
            for val, fps in sorted(agg.items(),
                                   key=lambda kv: -statistics.mean(kv[1])):
                lines.append(
                    f"  - `{val}`: mean={statistics.mean(fps):.2f}  "
                    f"max={max(fps):.2f}  n={len(fps)}"
                )
            lines.append("")
    # Failures summary
    if counts["oom"] or counts["too_few_frames"] or counts["error"]:
        lines.append("## Failure breakdown\n")
        bad = defaultdict(list)
        for r in rows:
            if r["status"] in ("oom", "too_few_frames", "error", "timeout"):
                bad[r["status"]].append(r["label"])
        for s, labels in bad.items():
            lines.append(f"### {s} ({len(labels)})")
            for lab in labels[:30]:
                lines.append(f"  - {lab}")
            if len(labels) > 30:
                lines.append(f"  - ... and {len(labels)-30} more")
            lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
