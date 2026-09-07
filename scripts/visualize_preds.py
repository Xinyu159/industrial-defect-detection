#!/usr/bin/env python3
"""Visual check of the stage-3 classifier -- NUMBERS ARE NOT EVIDENCE.

Re-trains the same SVM with the same split as train_classify.py and
renders ONE image with:
  - every test image the model got WRONG, labeled  GT -> predicted
  - 5 correct samples per class

so a human can eyeball whether the 96.9% is real. Mislabeling here is
usually the defect itself being ambiguous (the GT class still shows in
the image), not the model being broken.

Usage (from the repo root):
    python scripts/visualize_preds.py [--test-size 0.2] [--seed 20260907]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_classify as tc  # reuse load_rows / split / train exactly

OUT = tc.OUT
TILE = 200
STRIP = 26
GAP = 6
OK_GREEN = (60, 200, 60)
BAD_RED = (70, 70, 235)
CODES = [c for _, c, _ in tc.CLASSES]


def draw_text(img, s, org, scale=0.5, fill=(255, 255, 255)):
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        cv2.putText(img, s, (org[0] + dx, org[1] + dy),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (10, 10, 10), 1, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, fill,
                1, cv2.LINE_AA)


def tile(img_path, text, ok):
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    out = np.full((STRIP + TILE, TILE, 3), 25, np.uint8)
    if bgr is not None:
        out[STRIP:] = cv2.resize(bgr, (TILE, TILE))
    draw_text(out, text, (4, 18), 0.5, OK_GREEN if ok else BAD_RED)
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


def row_of(cells):
    """hstack cells with GAP between them, no trailing gap -> a row of n
    cells is n*TILE + (n-1)*GAP wide, matching the title bars."""
    row = cells[0]
    for c in cells[1:]:
        gap = np.full((c.shape[0], GAP, 3), 25, np.uint8)
        row = np.hstack([row, gap, c])
    return row


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

    # Build the page: title bars and tile rows, in order. Short rows are
    # padded right afterwards, so page width = widest content row.
    blocks = [("t", f"wrong ({len(wrong)} of {len(tei)}) -- "
                   "GT -> predicted, check by eye")]
    for k in range(0, len(wrong), 6):
        cells = [tile(f, f"GT {gt} -> {pr}", ok=False)
                 for i, f, gt, pr in wrong[k:k + 6]]
        blocks.append(("r", row_of(cells)))
    blocks.append(("t", f"correct samples, {args.correct_per_class} per class "
                        "(green = agrees with the XML ground truth)"))
    for code in CODES:
        got = [tei[j] for j in range(len(tei))
               if y[tei[j]] == code and pred[j] == code][:args.correct_per_class]
        blocks.append(("r", row_of([tile(files[k], code, ok=True) for k in got])))

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
