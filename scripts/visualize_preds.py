#!/usr/bin/env python3
"""Visual check of the stage-3 classifier -- NUMBERS ARE NOT EVIDENCE.

Re-trains the same SVM with the same split as train_classify.py and
renders ONE page laid out the way a compare sheet should be:
    TOP    = every test image the model got WRONG, one row per sample:
             [ truth: GT class, green boxes ] [ model: predicted class ]
    BOTTOM = correct samples per class (GT = prediction)

so a human can eyeball whether the 96.9% is real: look at the boxed
defect on the left and judge if the right-hand label is fair.

Usage (from the repo root):
    python scripts/visualize_preds.py [--test-size 0.2] [--seed 20260907]
"""
import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_classify as tc  # reuse load_rows / split / train exactly

OUT = tc.OUT
ROOT = Path(__file__).resolve().parent.parent
ANNS = ROOT / "data" / "NEU-DET" / "ANNOTATIONS"
TILE = 250  # upscaled so defects are actually visible
STRIP = 26
GAP = 6
OK_GREEN = (60, 200, 60)
BAD_RED = (70, 70, 235)
BOX_GREEN = (0, 230, 0)
CODES = [c for _, c, _ in tc.CLASSES]


def gt_boxes(stem):
    """VOC boxes of one image, scaled to TILE px (0-based)."""
    xml = ANNS / f"{stem}.xml"
    if not xml.exists():
        return []
    root = ET.parse(xml).getroot()
    s = TILE / 200.0
    out = []
    for o in root.findall("object"):
        bb = o.find("bndbox")
        x0, y0 = int(bb.findtext("xmin")) - 1, int(bb.findtext("ymin")) - 1
        x1, y1 = int(bb.findtext("xmax")), int(bb.findtext("ymax"))
        out.append((int(x0 * s), int(y0 * s), int(x1 * s), int(y1 * s)))
    return out


def draw_text(img, s, org, scale=0.5, fill=(255, 255, 255)):
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        cv2.putText(img, s, (org[0] + dx, org[1] + dy),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (10, 10, 10), 1, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, fill,
                1, cv2.LINE_AA)


def tile(img_path, text, color, boxed):
    """Upscaled original + label strip; GT truth gets green boxes, the
    model's view stays box-free so the human compares honestly."""
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        bgr = np.full((200, 200, 3), 200, np.uint8)
    bgr = cv2.resize(bgr, (TILE, TILE), interpolation=cv2.INTER_CUBIC)
    if boxed:
        for (x0, y0, x1, y1) in gt_boxes(Path(img_path).stem):
            cv2.rectangle(bgr, (x0, y0), (x1, y1), BOX_GREEN, 1)
    out = np.full((STRIP + TILE, TILE, 3), 25, np.uint8)
    out[STRIP:] = bgr
    draw_text(out, text, (4, 18), 0.5, color)
    return out


def title_bar(text, w):
    bar = np.full((34, w, 3), 25, np.uint8)
    draw_text(bar, text, (10, 24), 0.55)
    return bar


def pad_row(row, w):
    if row.shape[1] < w:
        pad = np.full((row.shape[0], w - row.shape[1], 3), 25, np.uint8)
        row = np.hstack([row, pad])
    return row


def hstack_gap(left, right):
    gap = np.full((left.shape[0], GAP, 3), 25, np.uint8)
    return np.hstack([left, gap, right])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=20260907)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--correct-per-class", type=int, default=5)
    args = p.parse_args()

    files, X, y = tc.load_rows()
    tri, tei = tc.split_idx(y, args.test_size, args.seed)
    acc, pred = tc.run(X[tri], X[tei], y[tri], y[tei], args.C, args.seed)

    # tei holds ABSOLUTE indices into X/y; pred is ordered like tei.
    wrong = [(tei[j], files[tei[j]], y[tei[j]], pred[j])
             for j in range(len(tei)) if y[tei[j]] != pred[j]]
    print(f"misclassified ({len(wrong)}):")
    for i, fpath, gt, pr in wrong:
        print(f"  {gt} -> {pr}   {Path(fpath).stem}.jpg")

    legend = ("codes: Cr=crazing | In=inclusion | Pa=patches | PS=pitted | "
              "RS=rolled-in | Sc=scratches")

    # --- TOP: wrong samples, one row each: [GT truth] [model prediction]
    blocks = [("t", legend),
              ("t", f"WRONG ({len(wrong)} of {len(tei)} test images) -- "
                    f"left = truth (green box)  right = what the model said")]
    for i, fpath, gt, pr in wrong:
        gt_tile = tile(fpath, f"truth: {gt}", OK_GREEN, boxed=True)
        pr_tile = tile(fpath, f"model: {pr}", BAD_RED, boxed=False)
        blocks.append(("r", hstack_gap(gt_tile, pr_tile)))

    # --- BOTTOM: correct samples per class, GT == prediction
    blocks.append(("t", f"CORRECT -- {args.correct_per_class} per class, "
                        "truth == prediction (green)"))
    for code in CODES:
        got = [tei[j] for j in range(len(tei))
               if y[tei[j]] == code and pred[j] == code][:args.correct_per_class]
        cells = [tile(files[k], f"{code} ok", OK_GREEN, boxed=True)
                 for k in got]
        for k in range(0, len(cells), 5):
            row = cells[k]
            for c in cells[k + 1:k + 5]:
                row = hstack_gap(row, c)
            blocks.append(("r", row))

    maxw = max(b[1].shape[1] for b in blocks if b[0] == "r")
    pieces = []
    for kind, payload in blocks:
        if kind == "t":
            pieces.append(title_bar(payload, maxw))
        else:
            pieces.append(pad_row(payload, maxw))
    page = np.vstack(pieces)
    out_png = OUT / "pred_check.png"
    cv2.imwrite(str(out_png), page)
    print(f"\nacc={acc:.1%} | 👉 打开查看: {out_png}")


if __name__ == "__main__":
    main()
