#!/usr/bin/env python3
"""Stage 3: classify the six defect classes from the C++ feature CSV.

Reads results/eval/features.csv (one row per image, produced by
`defect_detector features`), trains an RBF SVM on a fixed-seed 80/20
stratified split and reports:
  - test accuracy + per-class precision/recall/F1
  - a confusion matrix in ONE image (results/eval/confusion.png)
  - feature-group ablation: drop gray / GLCM / mask-group features and
    watch accuracy -- this tells which property separates which classes
    (GLCM texture should carry crazing, the pixel-segmentation loser)

Feature scaling (StandardScaler) is fit on the TRAIN split only --
fitting on the whole set would leak test statistics into training.
Each run appends one line to results/eval/eval_log.md.

Usage (from the repo root):
    python scripts/train_classify.py [--test-size 0.2] [--seed 20260907] [--C 1.0]
"""
import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "eval"
CSV_PATH = OUT / "features.csv"

CLASSES = [
    ("crazing", "Cr", "crazing"),
    ("inclusion", "In", "inclusion"),
    ("patches", "Pa", "patches"),
    ("pitted_surface", "PS", "pitted"),
    ("rolled-in_scale", "RS", "rolled-in"),
    ("scratches", "Sc", "scratches"),
]

# Feature column groups, by header name (order fixed by the C++ dumper).
GRAY = slice(1, 3)      # file, mean, std, ...
GLCM = slice(3, 7)
MASK = slice(7, 19)


def code_of(file_name):
    for prefix, code, _ in CLASSES:
        if file_name.startswith(prefix + "_"):
            return code
    return "??"


def draw_text(img, s, org, scale=0.5, fill=(255, 255, 255)):
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        cv2.putText(img, s, (org[0] + dx, org[1] + dy),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (15, 15, 15), 1, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, fill,
                1, cv2.LINE_AA)


def confusion_png(cm, acc, abla_note, out_path):
    """6x6 confusion matrix as one image; rows = true, cols = predicted.
    Diagonal cells get a green frame; deeper red = more samples."""
    codes = [c for _, c, _ in CLASSES]
    cell, label_w, title_h, head_h = 110, 150, 40, 30
    w = label_w + 6 * cell
    h = title_h + head_h + 6 * cell
    page = np.full((h, w, 3), 245, np.uint8)
    draw_text(page, f"SVM RBF test confusion  acc={acc:.1%}", (10, 26),
              0.55, fill=(15, 15, 15))
    draw_text(page, "row=true   col=pred", (w // 2 + 10, 26), 0.45,
              fill=(15, 15, 15))
    for ci, code in enumerate(codes):
        cx = label_w + ci * cell
        draw_text(page, code, (cx + cell // 2 - 10, head_h - 6),
                  fill=(15, 15, 15))
    cmax = max(1.0, cm.max())
    for ri, rcode in enumerate(codes):
        y = title_h + head_h + ri * cell
        draw_text(page, f"{rcode}", (label_w - 50, y + cell // 2 + 6), 0.55,
                  fill=(15, 15, 15))
        for ci in range(6):
            cx = label_w + ci * cell
            v = cm[ri, ci]
            cell_img = np.full((cell, cell, 3), 255, np.uint8)
            if v > 0:
                t = np.clip(v / cmax * 255, 0, 255)
                bgr = cv2.applyColorMap(np.uint8([t]), cv2.COLORMAP_TURBO)[0, 0]
                cell_img[:] = tuple(int(x) for x in bgr)
            page[y:y + cell, cx:cx + cell] = cell_img
            if v > 0:
                draw_text(page, str(int(v)),
                          (cx + cell // 2 - 14, y + cell // 2 + 6), 0.55)
            if ri == ci:
                cv2.rectangle(page, (cx + 1, y + 1), (cx + cell - 2, y + cell - 2),
                              (0, 200, 0), 2)
    footer = np.full((26, w, 3), 20, np.uint8)
    cv2.putText(footer, abla_note, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (220, 220, 220), 1, cv2.LINE_AA)
    page = np.vstack([page, footer])
    cv2.imwrite(str(out_path), page)
    return out_path


def load_rows():
    files, feats = [], []
    with open(CSV_PATH) as f:
        reader = csv.reader(f)
        header = next(reader)
        n_feat = len(header) - 1
        for row in reader:
            files.append(row[0])
            feats.append([float(x) for x in row[1:]])
    X = np.array(feats, dtype=np.float64)
    y = np.array([code_of(Path(p).stem) for p in files])
    assert X.shape[1] == n_feat
    return files, X, y


def split_idx(y, test_size, seed):
    """Stratified fixed-seed split, class by class."""
    rng = np.random.default_rng(seed)
    train, test = [], []
    for code in [c for _, c, _ in CLASSES]:
        idx = np.where(y == code)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_size))
        test.extend(idx[:n_test])
        train.extend(idx[n_test:])
    return np.array(train), np.array(test)


def run(X_tr, X_te, y_tr, y_te, C, seed):
    sc = StandardScaler().fit(X_tr)  # train-only statistics!
    svm = SVC(kernel="rbf", C=C, gamma="scale", random_state=seed)
    svm.fit(sc.transform(X_tr), y_tr)
    pred = svm.predict(sc.transform(X_te))
    return accuracy_score(y_te, pred), pred


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=20260907)
    p.add_argument("--C", type=float, default=1.0)
    args = p.parse_args()

    files, X, y = load_rows()
    train_idx, test_idx = split_idx(y, args.test_size, args.seed)
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    counts = defaultdict(int)
    for c in y:
        counts[c] += 1
    print("samples:", dict(counts), "| train", len(train_idx), "test",
          len(test_idx))

    acc, pred = run(X_tr, X_te, y_tr, y_te, args.C, args.seed)
    codes = [c for _, c, _ in CLASSES]
    print(f"\ntest acc (all {X.shape[1]} features, RBF C={args.C}): "
          f"{acc:.1%}\n")
    print(classification_report(y_te, pred, labels=codes, digits=3,
                                zero_division=0))

    # Feature-group ablation: the accuracy cost of removing each group.
    ablation = {"full": acc}
    groups = {"-gray": GRAY, "-glcm": GLCM, "-mask": MASK}
    for name, sl in groups.items():
        keep = [i for i in range(X.shape[1]) if i not in range(*sl.indices(X.shape[1]))]
        a, _ = run(X_tr[:, keep], X_te[:, keep], y_tr, y_te, args.C, args.seed)
        ablation[name] = a
    acc_g, _ = run(X_tr[:, GLCM], X_te[:, GLCM], y_tr, y_te, args.C, args.seed)
    ablation["glcm-only"] = acc_g
    print("ablation (loss = full acc - this acc, i.e. what removing the "
          "group costs):")
    for name, a in ablation.items():
        if name == "full":
            print(f"  {name:<10} {a:6.1%}")
            continue
        print(f"  {name:<10} {a:6.1%}   loss {acc - a:+.1%}")

    cm = confusion_matrix(y_te, pred, labels=codes)
    note = (f"ablation: -gray {ablation['-gray']:.0%}  -glcm "
            f"{ablation['-glcm']:.0%}  -mask {ablation['-mask']:.0%} | "
            f"test {len(test_idx)} imgs | seed={args.seed} C={args.C}")
    png = confusion_png(cm, acc, note, OUT / "confusion.png")

    log = OUT / "eval_log.md"
    if not log.exists():
        log.write_text("# stage-3 classification runs\n\n")
    with open(log, "a") as f:
        ab = " ".join(f"{n}={a:.3f}" for n, a in ablation.items())
        f.write(f"- {datetime.now():%Y-%m-%d %H:%M} | acc={acc:.3f} | {ab} | "
                f"test={len(test_idx)} seed={args.seed} C={args.C}\n")

    print("\n👉 打开查看: " + str(png))
    print("👉 CSV 输入 : " + str(CSV_PATH))
    print("👉 参数日志 : " + str(log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
