#!/usr/bin/env python3
"""Stage 4: candidate-box detection -- the classifier now answers WHERE.

An image-level classifier cannot draw its own boxes, so this turns the
task into region-proposal + box classification:
  TRAIN   every GT box is a positive patch (labeled with its class),
          random non-overlapping patches are BACKGROUND negatives
          -> 7-class SVM (6 defects + BG) on box-level features
          (C++ `patchfeat` computes features for arbitrary patches)
  INFER   test images are segmented with the three stage-2 strategies,
          every connected component becomes a candidate box, candidates
          are classified, BG ones are dropped -- the survivors ARE the
          model's detections (box + class)
  JUDGE   predicted boxes are matched to GT boxes by IoU >= 0.5 AND
          equal class: TP; unmatched GT = FN; unmatched pred = FP.
          Per-class precision/recall/F1 are printed and EVERY test frame
          gets ONE png -- left truth boxed, right the model's own boxes,
          green = hit, red = missed GT / false alarm -- stored in
          results/eval/detect/good/ and .../bad/ so the numbers can be
          eyeballed one sample at a time. detect_log.md records each run.

Usage (from the repo root):
    python scripts/detect_boxes.py [--seed 20260907] [--C 1.0] [--iou 0.5]
"""
import argparse
import subprocess
from datetime import datetime
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
BINARY = ROOT / "build" / "defect_detector"
IMGS = ROOT / "data" / "NEU-DET" / "IMAGES"
ANNS = ROOT / "data" / "NEU-DET" / "ANNOTATIONS"
OUT = ROOT / "results" / "eval"
TMP = Path("/tmp/idd_detect")

CLASSES = [
    ("crazing", "Cr", "crazing"),
    ("inclusion", "In", "inclusion"),
    ("patches", "Pa", "patches"),
    ("pitted_surface", "PS", "pitted"),
    ("rolled-in_scale", "RS", "rolled-in"),
    ("scratches", "Sc", "scratches"),
]
BG = "BG"
CODES = [c for _, c, _ in CLASSES]
IMG = 200  # NEU frames are 200x200

TILE = 220
STRIP = 26
GAP = 6
GREEN = (60, 200, 60)
RED = (70, 70, 235)
BOX_TP = (60, 220, 60)
BOX_BAD = (70, 70, 235)  # FN on truth tiles, FP on model tiles


# ---------------- boxes ----------------

def gts_of(path):
    """[(box, class)] with box as 0-based HALF-OPEN slice (x0,y0,x1,y1)."""
    xml = ANNS / (Path(path).stem + ".xml")
    out = []
    if not xml.exists():
        return out
    root = ET.parse(xml).getroot()
    for o in root.findall("object"):
        bb = o.find("bndbox")
        name = o.findtext("name", "")
        code = next((c for p, c, _ in CLASSES if p == name), "??")
        out.append(((int(bb.findtext("xmin")) - 1, int(bb.findtext("ymin")) - 1,
                     int(bb.findtext("xmax")), int(bb.findtext("ymax"))), code))
    return out


def area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    uni = area(a) + area(b) - inter
    return inter / uni if uni else 0.0


def nms(boxes, thr=0.4):
    """Greedy NMS over (x0,y0,x1,y1): biggest first, drop heavy overlaps."""
    kept = []
    for b in sorted(boxes, key=area, reverse=True):
        if not any(iou(b, k) > thr for k in kept):
            kept.append(b)
    return kept


def bg_patches(gt_boxes, n, rng):
    """n random patches overlapping NO GT box (clean-steel negatives)."""
    out = []
    for _ in range(n):
        for _ in range(60):
            w = int(rng.integers(24, 90))
            h = int(rng.integers(24, 90))
            x = int(rng.integers(0, IMG - w + 1))
            y = int(rng.integers(0, IMG - h + 1))
            cand = (x, y, x + w, y + h)
            if all(iou(cand, g) == 0.0 for g in gt_boxes):
                out.append(cand)
                break
    return out


def clamp_box(b, lo=0, hi=IMG - 1):
    """Mirror the C++ clamp in patchfeat: the CSV key is the box AFTER
    clamping into the image, not the one we asked for."""
    return tuple(max(lo, min(v, hi)) for v in b)


def run_patchfeat(lines, csv_path):
    """lines: [(path, x0, y0, x1, y1)] -> {(path,*clamped_box): feature}.
    Lookup keys must be clamped exactly like the C++ side does."""
    list_path = TMP / "patches.txt"
    list_path.write_text("\n".join(f"{p} {x0} {y0} {x1} {y1}"
                                   for p, x0, y0, x1, y1 in lines) + "\n")
    subprocess.run([str(BINARY), "patchfeat", str(list_path), str(csv_path)],
                   check=True)
    feats = {}
    with open(csv_path) as f:
        f.readline()  # header
        for ln in f:
            c = ln.rstrip("\n").split(",")
            feats[(c[0], int(c[1]), int(c[2]), int(c[3]), int(c[4]))] = \
                np.array([float(v) for v in c[5:]])
    return feats


def lookup_feat(feats, line):
    """Feature row for (path, x0, y0, x1, y1) under the clamped key."""
    return feats[(line[0], *clamp_box(line[1:]))]


def candidates(path, masks_dir, min_area=12):
    """Connected components of the three masks -> candidate boxes."""
    stem = Path(path).stem
    boxes = []
    for tag in ("otsu", "edge", "hat"):
        m = cv2.imread(str(masks_dir / f"{stem}_{tag}_mask.png"),
                       cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        _, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, labels.max() + 1):
            a = int(stats[i, cv2.CC_STAT_AREA])
            if a < min_area:
                continue
            x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
            w, h = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
            # patchfeat refuses patches thinner than 4 px either way
            # (checked AFTER clamping into the image, which shrinks boxes
            # touching the edge) -- such slivers could never be
            # classified, drop them here so every candidate below is
            # actually classifiable.
            cb = clamp_box((x, y, x + w, y + h))
            if cb[2] - cb[0] < 4 or cb[3] - cb[1] < 4:
                continue
            boxes.append((x, y, x + w, y + h))
    return nms(boxes)[:60]


def match_boxes(gts, preds, iou_thr):
    """gts/preds: [(box, cls)]. -> tp: (gi, pi, cls); wrong: (gi, pi);
    fn: [gi]; fp: [pi]. A pred that overlaps a GT but says another class
    is counted as wrong: GT -> FN and box -> FP (marked red)."""
    tps, wrong = [], []
    scored = [(iou(gb, pb), gi, pi)
              for gi, (gb, gc) in enumerate(gts)
              for pi, (pb, pc) in enumerate(preds)
              if gc == pc and iou(gb, pb) >= iou_thr]
    scored.sort(reverse=True)
    used_g, used_p = set(), set()
    for _, gi, pi in scored:
        if gi not in used_g and pi not in used_p:
            used_g.add(gi)
            used_p.add(pi)
            tps.append((gi, pi, gts[gi][1]))
    for gi, (gb, gc) in enumerate(gts):
        if gi in used_g:
            continue
        for pi, (pb, pc) in enumerate(preds):
            if pi in used_p or iou(gb, pb) < iou_thr:
                continue
            used_g.add(gi)
            used_p.add(pi)
            wrong.append((gi, pi))
            break
    fn = [gi for gi in range(len(gts)) if gi not in used_g]
    fp = [pi for pi in range(len(preds)) if pi not in used_p]
    return tps, wrong, fn, fp


# ---------------- rendering ----------------

def draw_text(img, s, org, scale=0.5, fill=(255, 255, 255)):
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        cv2.putText(img, s, (org[0] + dx, org[1] + dy),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (10, 10, 10), 1, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, fill,
                1, cv2.LINE_AA)


def det_tile(path, colored_boxes, text, tcolor):
    """Original, upscaled, with (box, BGR color) rectangles on it."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        bgr = np.full((IMG, IMG, 3), 180, np.uint8)
    bgr = cv2.resize(bgr, (TILE, TILE), interpolation=cv2.INTER_CUBIC)
    s = TILE / IMG
    for (x0, y0, x1, y1), color in colored_boxes:
        cv2.rectangle(bgr, (int(x0 * s), int(y0 * s)),
                      (int(x1 * s), int(y1 * s)), color, 2)
    out = np.full((STRIP + TILE, TILE, 3), 25, np.uint8)
    out[STRIP:] = bgr
    draw_text(out, text, (4, 18), 0.5, tcolor)
    return out


def hcat(a, b):
    gap = np.full((a.shape[0], GAP, 3), 25, np.uint8)
    return np.hstack([a, gap, b])


def truth_tile(path, gts, preds, iou_thr, tag):
    """Truth side: green = GT hit by a same-class box; red = GT not hit
    (missed, or only overlapped by a wrong-class box)."""
    tps, wrong, fn, fp = match_boxes(gts, preds, iou_thr)
    missed = [gi for gi, _ in wrong] + fn
    boxes = [(gts[gi][0], BOX_BAD) for gi in missed]
    boxes += [(gts[t[0]][0], BOX_TP) for t in tps]
    return det_tile(path, boxes, tag, GREEN)


def model_tile(path, gts, preds, iou_thr, tag="model"):
    """Model side: green = its detection hit the truth; red = false
    alarms + boxes whose class contradicts the frame's GT class."""
    tps, wrong, fn, fp = match_boxes(gts, preds, iou_thr)
    false = [pi for _, pi in wrong] + fp
    boxes = [(preds[pi][0], BOX_BAD) for pi in false]
    boxes += [(preds[t[1]][0], BOX_TP) for t in tps]
    return det_tile(path, boxes, tag, GREEN if not false else RED)


# ---------------- main ----------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=20260907)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--bg-per-img", type=int, default=2)
    p.add_argument("--bg-weight", type=float, default=1.0,
                   help="class_weight of BG relative to defect classes. At "
                        "inference BG is the overwhelming majority of "
                        "candidates; training it with weight 1.0 makes the "
                        "SVM fire defect boxes everywhere (observed: 45%% "
                        "non-BG). Raise it to push the boundary toward BG.")
    p.add_argument("--frame-gate", type=int, default=1,
                   help="two-stage coarse-to-fine check used in industry: "
                        "the image-level classifier (stage 3, ~97% acc) "
                        "first says what class this frame can be, then only "
                        "candidate boxes OF THAT class survive. Kills "
                        "cross-class false alarms; costs the frame whenever "
                        "the coarse stage is wrong.")
    p.add_argument("--min-area", type=int, default=12)
    args = p.parse_args()

    TMP.mkdir(parents=True, exist_ok=True)
    files = sorted(str(f) for f in IMGS.glob("*.jpg"))
    y = np.array([next(c for p_, c, _ in CLASSES
                       if Path(f).name.startswith(p_ + "_")) for f in files])

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import train_classify as tc
    tri, tei = tc.split_idx(y, args.test_size, args.seed)
    print(f"images: train {len(tri)} test {len(tei)}")

    # ---- train patches: GT boxes = positives, random clean = BG ----
    rng = np.random.default_rng(args.seed)
    train_lines, train_labels = [], []
    for i in tri:
        path = files[i]
        gt_pairs = gts_of(path)
        for b, c in gt_pairs:
            if area(b) >= 16:
                train_lines.append((path, *b))
                train_labels.append(c)
        for b in bg_patches([g for g, _ in gt_pairs], args.bg_per_img, rng):
            train_lines.append((path, *b))
            train_labels.append(BG)
    n_pos = train_labels.count(BG)
    print(f"train patches: {len(train_lines)} ({len(train_labels) - n_pos} "
          f"positive, {n_pos} background)")

    print("extracting train patch features...")
    tr_feats = run_patchfeat(train_lines, TMP / "feats_train.csv")
    X_tr = np.array([lookup_feat(tr_feats, l) for l in train_lines])
    y_tr = np.array(train_labels)

    # ---- masks of test frames -> candidate boxes ----
    masks_dir = TMP / "masks"
    test_list = TMP / "test_imgs.txt"
    test_list.write_text("\n".join(files[i] for i in tei) + "\n")
    print("segmenting test frames...")
    subprocess.run([str(BINARY), "batch", str(test_list), str(masks_dir)],
                   check=True)

    cand_by_img = {i: [] for i in tei}
    for i in tei:
        cand_by_img[i] = candidates(files[i], masks_dir, args.min_area)
    n_cand = sum(len(v) for v in cand_by_img.values())
    print(f"candidates: {n_cand} from {len(tei)} test images")

    print("extracting candidate features...")
    te_feats = run_patchfeat([(files[i], *b) for i in tei
                              for b in cand_by_img[i]], TMP / "feats_cand.csv")

    # ---- train the box classifier (scaler on train only) ----
    w = {c: 1.0 for c in CODES}
    w[BG] = args.bg_weight
    sc = StandardScaler().fit(X_tr)
    svm = SVC(kernel="rbf", C=args.C, gamma="scale", class_weight=w,
              random_state=args.seed)
    svm.fit(sc.transform(X_tr), y_tr)

    # ---- frame gate: image-level classifier decides the allowed class ----
    pic = {}
    if args.frame_gate:
        csv_paths, X_img, y_img = tc.load_rows()
        # features.csv rows carry repo-relative paths; key by stem instead.
        by_stem = {Path(p).stem: k for k, p in enumerate(csv_paths)}
        assert len(csv_paths) == len(files), \
            f"image features cover {len(csv_paths)} files, {len(files)} expected"
        tr_rows = [by_stem[Path(files[i]).stem] for i in tri]
        te_rows = [by_stem[Path(files[i]).stem] for i in tei]
        _, pred_img = tc.run(X_img[tr_rows], X_img[te_rows], y_img[tr_rows],
                             y_img[te_rows], args.C, args.seed)
        for j, i in enumerate(tei):
            pic[i] = pred_img[j]
        wrong_gate = sum(1 for i in tei if pic[i] != y[i])
        print(f"frame gate: image classifier acc {1 - wrong_gate / len(tei):.1%} "
              f"on this split")

    # ---- classify candidates: keep the SVM's own class per box ----
    preds_by_img = {}
    n_pred = 0
    n_gated = 0
    for i in tei:
        out = []
        for b in cand_by_img[i]:
            feat = lookup_feat(te_feats, (files[i], *b))
            cls = svm.predict(sc.transform(feat.reshape(1, -1)))[0]
            if cls != BG and (not args.frame_gate or cls == pic[i]):
                out.append((b, cls))
                n_pred += 1
            elif args.frame_gate and cls != BG and cls != pic[i]:
                n_gated += 1
        preds_by_img[i] = out
    print(f"detections (non-BG): {n_pred}"
          + (f" | cross-class boxes gated out: {n_gated}" if args.frame_gate
             else ""))

    all_tp, all_fp, all_fn = defaultdict(int), defaultdict(int), defaultdict(int)
    bad, good = [], []
    for i in tei:
        gts = gts_of(files[i])
        preds = preds_by_img[i]
        tps, wrong, fn, fp = match_boxes(gts, preds, args.iou)
        for _, _, c in tps:
            all_tp[c] += 1
        for gi in fn:
            all_fn[gts[gi][1]] += 1
        for pi in fp:
            all_fp[preds[pi][1]] += 1
        for gi, pi in wrong:
            all_fn[gts[gi][1]] += 1
            all_fp[preds[pi][1]] += 1
        (bad if (fn or fp or wrong) else good).append(i)

    print(f"\n{'class':<6}{'TP':>5}{'FP':>5}{'FN':>5}  {'precision':>10}"
          f"{'recall':>8}{'f1':>7}")
    sum_p = sum_r = 0.0
    for c in CODES:
        tp, fp, fn = all_tp[c], all_fp[c], all_fn[c]
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        sum_p += pr
        sum_r += rc
        print(f"{c:<6}{tp:>5}{fp:>5}{fn:>5}  {pr:>10.1%}{rc:>8.1%}{f1:>7.1%}")
    tp, fp, fn = sum(all_tp.values()), sum(all_fp.values()), sum(all_fn.values())
    print(f"\noverall: P={tp / (tp + fp):.1%} R={tp / (tp + fn):.1%} "
          f"(macro P={sum_p / 6:.1%} R={sum_r / 6:.1%}) | frames: "
          f"bad {len(bad)} good {len(good)}")

    # ---- per-sample pages: ONE frame per PNG so each is checkable
    # against its own original. results/eval/detect/bad/ and .../good/.
    vis_root = OUT / "detect"
    good_set = set(good)
    for sub in ("bad", "good"):
        (vis_root / sub).mkdir(parents=True, exist_ok=True)
    n_saved = 0
    for i in tei:
        gts = gts_of(files[i])
        preds = preds_by_img[i]
        tag_m = "model"
        if args.frame_gate:
            tag_m += f" ({pic[i]})"
        pair = hcat(truth_tile(files[i], gts, preds, args.iou,
                               f"truth {y[i]}"),
                    model_tile(files[i], gts, preds, args.iou, tag_m))
        sub = "good" if i in good_set else "bad"
        cv2.imwrite(str(vis_root / sub / f"{Path(files[i]).stem}.png"), pair)
        n_saved += 1
    print(f"\n👉 打开查看: {vis_root}/  ({len(bad)} bad + {len(good)} good, "
          f"{n_saved} frames)")

    log = OUT / "detect_log.md"
    if not log.exists():
        log.write_text("# stage-4 box-level detection runs\n\n")
    with open(log, "a") as f:
        f.write(f"- {datetime.now():%Y-%m-%d %H:%M} | P={tp/(tp+fp):.3f} "
                f"R={tp/(tp+fn):.3f} | macro P={sum_p/6:.3f} R={sum_r/6:.3f} "
                f"| good {len(good)}/{len(tei)} frames | bgW={args.bg_weight} "
                f"bgN={args.bg_per_img} gate={bool(args.frame_gate)} "
                f"iou={args.iou} seed={args.seed} C={args.C}\n")
    print("👉 参数日志 : " + str(log))


if __name__ == "__main__":
    main()
