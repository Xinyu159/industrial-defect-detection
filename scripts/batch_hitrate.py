#!/usr/bin/env python3
"""Batch hit-rate evaluation of the C++ segmentation strategies.

Sample N images per defect class, segment each with all three strategies
(via `defect_detector batch`), then judge the masks against the XML
ground-truth boxes the way a production line judges a detector:

    a GT box is HIT when >= --hit-frac of its pixels are foreground.

Pixel-level overlap is deliberately NOT required: NEU boxes are region
declarations ("defects live in here"), not pixel truth -- exactly like
the operator windows in a real surface-inspection system.

Outputs (fixed filenames; the run parameters go to the log instead of
multiplying files):
    results/eval/hitrate.png   -- one image: hit-rate + over-seg matrix
    results/eval/hitrate.csv   -- machine-readable table
    results/eval/batch_log.md  -- one appended line per run
    results/eval/masks/        -- C++ masks of the sampled frames

Usage (from the repo root):
    python scripts/batch_hitrate.py --per-class 10 [--hit-frac 0.01]
"""
import argparse
import csv
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IMGS = ROOT / "data" / "NEU-DET" / "IMAGES"
ANNS = ROOT / "data" / "NEU-DET" / "ANNOTATIONS"
BINARY = ROOT / "build" / "defect_detector"
OUT = ROOT / "results" / "eval"

CLASSES = [  # NEU-DET prefix -> short code -> English name
    ("crazing", "Cr", "crazing"),
    ("inclusion", "In", "inclusion"),
    ("patches", "Pa", "patches"),
    ("pitted_surface", "PS", "pitted"),
    ("rolled-in_scale", "RS", "rolled-in"),
    ("scratches", "Sc", "scratches"),
]
STRATEGIES = [("otsu", "Otsu"), ("edge", "Edge"), ("hat", "Black-hat")]


def boxes_of(xml_path):
    """VOC bndbox -> list of (x0, y0, x1, y1) 0-indexed, clamped to the
    200x200 image so the ROI slice is always inside."""
    import xml.etree.ElementTree as ET

    root = ET.parse(xml_path).getroot()
    out = []
    for o in root.findall("object"):
        bb = o.find("bndbox")
        x0, y0 = int(bb.findtext("xmin")) - 1, int(bb.findtext("ymin")) - 1
        x1, y1 = int(bb.findtext("xmax")), int(bb.findtext("ymax"))
        out.append((max(0, x0), max(0, y0), min(200, x1), min(200, y1)))
    return out


def sample_frames(per_class, seed):
    """Fixed-seed sample: N images per class -> {cls: [paths]}."""
    files = sorted(IMGS.glob("*.jpg"))
    by_class = {prefix: [] for prefix, _, _ in CLASSES}
    for f in files:
        # prefix match must handle compound names: pitted_surface_*,
        # rolled-in_scale_* -- split("_")[0] would give "pitted"/"rolled".
        for prefix in by_class:
            if f.name.startswith(prefix + "_"):
                by_class[prefix].append(f)
                break
    rng = random.Random(seed)
    return {c: rng.sample(by_class[c], min(per_class, len(by_class[c])))
            for c in by_class}


def run_batch(manifest, masks_dir):
    masks_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(BINARY), "batch", str(manifest), str(masks_dir)],
                   check=True)
    return masks_dir


def accumulate(img_cls_pairs, masks_dir):
    """Per (class, strategy): hits, boxes, global fg % mean.

    Hit rule (relative density -- an absolute share alone is worthless:
    a detector that flags 50% of every frame trivially hits every box):
    a GT box is hit when its fg share reaches
        max(hit_frac, hit_gain * global_fg_share)
    i.e. defects are clearly denser inside the box than in the frame.
    """
    acc = {(c, s): {"hits": 0, "fg_sum": 0.0, "n_img": 0}
           for c, _, _ in CLASSES for s, _ in STRATEGIES}
    boxes_total = {c: 0 for c, _, _ in CLASSES}
    for cls, img_path in img_cls_pairs:
        stem = img_path.stem
        xml_path = ANNS / f"{stem}.xml"
        boxes = boxes_of(xml_path) if xml_path.exists() else []
        boxes_total[cls] += len(boxes)
        for strat, _ in STRATEGIES:
            mask = cv2.imread(str(masks_dir / f"{stem}_{strat}_mask.png"),
                              cv2.IMREAD_GRAYSCALE)
            fg_ratio = cv2.countNonZero(mask) / float(mask.size)
            key = (cls, strat)
            acc[key]["fg_sum"] += fg_ratio
            acc[key]["n_img"] += 1
            thr = max(hit_frac, hit_gain * fg_ratio)
            for (x0, y0, x1, y1) in boxes:
                roi = mask[y0:y1, x0:x1]
                if cv2.countNonZero(roi) >= thr * roi.size:
                    acc[key]["hits"] += 1
    return acc, boxes_total


def ascii_table(acc, boxes_total):
    lines = [f"{'class':<14}{'boxes':>6} | " +
             " ".join(f"{'hits':>7}{'fg%':>7}" for _, _ in STRATEGIES)]
    for prefix, code, _ in CLASSES:
        cells = []
        for strat, _ in STRATEGIES:
            a = acc[(prefix, strat)]
            mean_fg = 100.0 * a["fg_sum"] / max(1, a["n_img"])
            cells.append(f"{a['hits']:>3}/{boxes_total[prefix]:<3}{mean_fg:>7.2f}")
        lines.append(f"{code:<14}{boxes_total[prefix]:>6} | " + " | ".join(cells))
    return "\n".join(lines)


def best_per_class(acc, boxes_total):
    best = []
    for prefix, code, _ in CLASSES:
        total = boxes_total[prefix]
        if total == 0:
            continue
        ranked = []
        for strat, _ in STRATEGIES:
            a = acc[(prefix, strat)]
            rate = a["hits"] / total
            mean_fg = 100.0 * a["fg_sum"] / max(1, a["n_img"])
            ranked.append((rate, -mean_fg, strat))  # tie: lower fg wins
        ranked.sort(reverse=True)
        rate, nfg, strat = ranked[0]
        best.append((code, total, strat, acc[(prefix, strat)]["hits"], rate, -nfg))
    return best


def draw_text(img, s, org, scale=0.5, fill=(255, 255, 255)):
    """putText with a cheap dark outline so numbers stay readable on any
    colormap background."""
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        cv2.putText(img, s, (org[0] + dx, org[1] + dy),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (15, 15, 15), 1, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, fill,
                1, cv2.LINE_AA)


def matrix_panel(title, rows, invert=False):
    """rows: list of 3 floats (one per strategy). Returns a BGR image with a
    colored cell per value. invert=True -> low value is good (red)."""
    cell_w, cell_h, label_w = 170, 64, 210
    header_h, title_h = 26, 34
    w = label_w + 3 * cell_w
    h = title_h + header_h + len(rows) * cell_h
    panel = np.full((h, w, 3), 245, np.uint8)
    cv2.rectangle(panel, (0, 0), (w - 1, h - 1), (120, 120, 120), 1)
    draw_text(panel, title, (8, 22), 0.55, fill=(15, 15, 15))
    for ci, (_, name) in enumerate(STRATEGIES):
        cx = label_w + ci * cell_w
        draw_text(panel, name, (cx + cell_w // 2 - 30, header_h - 7),
                  fill=(15, 15, 15))
    for ri, (code, name, vals) in enumerate(rows):
        y = title_h + header_h + ri * cell_h
        draw_text(panel, f"{code}  {name}", (6, y + 40), 0.5, fill=(15, 15, 15))
        for ci, v in enumerate(vals):
            cx = label_w + ci * cell_w
            t = v if not invert else 1.0 - min(1.0, v / 0.05)
            cell = np.full((cell_h, cell_w, 3), 255, np.uint8)
            bgr = cv2.applyColorMap(np.uint8([np.clip(t * 255, 0, 255)]),
                                    cv2.COLORMAP_TURBO)[0, 0]
            cell[:] = tuple(int(x) for x in bgr)
            panel[y:y + cell_h, cx:cx + cell_w] = cell
            s = f"{100.0 * v:.0f}%" if v < 1.0 else f"{v:.1f}%"
            draw_text(panel, s, (cx + cell_w // 2 - 22, y + cell_h // 2 + 6), 0.6)
    return panel


def make_png(acc, boxes_total, best, params, out_path):
    rows_hit, rows_fg = [], []
    for prefix, code, name in CLASSES:
        total = boxes_total[prefix]
        hit = [acc[(prefix, s)]["hits"] / total if total else 0.0
               for s, _ in STRATEGIES]
        fg = [100.0 * acc[(prefix, s)]["fg_sum"] / max(1, acc[(prefix, s)]["n_img"])
              for s, _ in STRATEGIES]
        rows_hit.append((code, name, hit))
        rows_fg.append((code, name, fg))
    p_hit = matrix_panel("GT-box hit rate   (red=high = defects found)", rows_hit)
    p_fg = matrix_panel("mean global fg %   (red=low = less over-seg)", rows_fg,
                        invert=True)
    gap = np.full((12, p_hit.shape[1], 3), 245, np.uint8)
    page = np.vstack([p_hit, gap, p_fg])
    note = (f"N={params.per_class}/class | hit = box fg >= max("
            f"{100 * params.hit_frac:.0f}%, {params.hit_gain:.1f}x frame fg) | "
            f"seed={params.seed} | {datetime.now():%Y-%m-%d %H:%M}")
    footer = np.full((26, page.shape[1], 3), 20, np.uint8)
    cv2.putText(footer, note, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (230, 230, 230), 1, cv2.LINE_AA)
    page = np.vstack([page, footer])
    cv2.imwrite(str(out_path), page)
    return out_path


def append_log(best, params):
    log = OUT / "batch_log.md"
    if not log.exists():
        log.write_text("# batch hit-rate runs\n\n"
                       "one line per run: params | best pick per class\n\n")
    picks = " | ".join(f"{c}:{s}({100 * r:.0f}%)" for c, _, s, _, r, _ in best)
    line = (f"- {datetime.now():%Y-%m-%d %H:%M} | N={params.per_class} "
            f"gain={params.hit_gain}x | hit_frac>={100 * params.hit_frac:.1f}%"
            f" | seed={params.seed} | {picks}\n")
    with open(log, "a") as f:
        f.write(line)
    return log


def main():
    global hit_frac, hit_gain
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-class", type=int, default=10)
    p.add_argument("--hit-frac", type=float, default=0.01,
                   help="floor: min fg share inside a GT box to count as hit")
    p.add_argument("--hit-gain", type=float, default=2.0,
                   help="box fg share must reach gain x global fg share")
    p.add_argument("--seed", type=int, default=20260907)
    args = p.parse_args()
    hit_frac = args.hit_frac
    hit_gain = args.hit_gain

    OUT.mkdir(parents=True, exist_ok=True)
    masks_dir = OUT / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    sampled = sample_frames(args.per_class, args.seed)
    pairs = [(cls, path) for cls, paths in sampled.items() for path in paths]
    manifest = masks_dir / "manifest.txt"
    manifest.write_text("\n".join(str(p) for _, p in pairs) + "\n")
    print(f"sampled {len(pairs)} images ({args.per_class}/class, "
          f"seed={args.seed}) -> {manifest}")

    run_batch(manifest, masks_dir)
    acc, boxes_total = accumulate(pairs, masks_dir)

    print(f"\nhit rule: box fg share >= max({100 * args.hit_frac:.0f}%, "
          f"{args.hit_gain:.1f}x frame fg share)\n")
    print(ascii_table(acc, boxes_total))
    best = best_per_class(acc, boxes_total)
    print("\nbest per class (tie -> lower fg):")
    for code, total, strat, hits, rate, fg in best:
        print(f"  {code:<4} {strat:<6} {hits:>3}/{total:<3}  hit={100 * rate:5.1f}%"
              f"  fg={fg:.2f}%")

    csv_path = OUT / "hitrate.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "strategy", "hits", "boxes", "hit_rate",
                    "mean_global_fg_pct"])
        for prefix, code, _ in CLASSES:
            for strat, _ in STRATEGIES:
                a = acc[(prefix, strat)]
                w.writerow([code, strat, a["hits"], boxes_total[prefix],
                            a["hits"] / boxes_total[prefix] if boxes_total[prefix] else 0.0,
                            round(100 * a["fg_sum"] / max(1, a["n_img"]), 2)])

    png_path = make_png(acc, boxes_total, best, args, OUT / "hitrate.png")
    log_path = append_log(best, args)

    print("\n👉 打开查看: " + str(png_path))
    print("👉 CSV     : " + str(csv_path))
    print("👉 参数日志 : " + str(log_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
