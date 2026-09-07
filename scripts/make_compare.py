#!/usr/bin/env python3
"""Single-page comparison sheets: GT (green boxes from VOC xml) side by side
with every segmentation strategy overlay, all in ONE image per sample, plus
one combined page for all samples.

Usage:
    python scripts/make_compare.py <out_prefix> <neu_img> [<out_prefix> <neu_img> ...]

Example:
    python scripts/make_compare.py \\
        results/seg_rs1 data/NEU-DET/IMAGES/rolled-in_scale_1.jpg \\
        results/seg_sc1 data/NEU-DET/IMAGES/scratches_1.jpg
        # expects files: results/seg_rs1_{otsu,edge,hat}_overlay.png

Outputs (convention: everything comparable lands in a single file):
    results/compare/<img_stem>_compare.png   -- per-sample sheet
    results/compare/all_compare.png          -- all samples stacked
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

STRATEGIES = [("otsu", "Otsu"), ("edge", "Edge"), ("hat", "Black-hat")]
GREEN = (0, 230, 0)
TILE = 200
STRIP = 24  # label strip on top of each tile
GAP = 3

CODE = {
    "crazing": "Cr", "inclusion": "In", "patches": "Pa",
    "pitted_surface": "PS", "rolled-in_scale": "RS", "scratches": "Sc",
}


def boxes_of(xml_path):
    root = ET.parse(xml_path).getroot()
    out = []
    for o in root.findall("object"):
        bb = o.find("bndbox")
        out.append((int(bb.findtext("xmin")), int(bb.findtext("ymin")),
                    int(bb.findtext("xmax")), int(bb.findtext("ymax"))))
    return out


def code_of(xml_path):
    root = ET.parse(xml_path).getroot()
    name = root.findtext("object/name", "")
    return CODE.get(name, name[:2].upper())


def tile(gray, text, draw_boxes=None, color=GREEN):
    """200x200 gray tile + 24px label strip, optional green boxes."""
    h, w = gray.shape
    out = np.full((STRIP + h, w, 3), 255, np.uint8)
    if draw_boxes:
        for b in draw_boxes:
            cv2.rectangle(out[STRIP:], (b[0], b[1]), (b[2], b[3]), color, 2)
    else:
        out[STRIP:] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.putText(out, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)
    return out


def load_optional(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return img if img is not None else None


def main():
    args = sys.argv[1:]
    if len(args) < 2 or len(args) % 2 != 0:
        print(__doc__)
        return 1
    pairs = [(Path(args[i]), Path(args[i + 1])) for i in range(0, len(args), 2)]

    out_dir = Path("results/compare")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for prefix, img_path in pairs:
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"!! cannot read: {img_path}")
            continue
        xml_path = img_path.parent.parent / "ANNOTATIONS" / (img_path.stem + ".xml")
        boxes = boxes_of(xml_path) if xml_path.exists() else []
        code = code_of(xml_path) if xml_path.exists() else "??"

        cells = [tile(gray, f"GT {code} (truth)", draw_boxes=boxes)]
        for strat, label in STRATEGIES:
            ov = load_optional(f"{prefix}_{strat}_overlay.png")
            if ov is None:
                ov = np.full_like(gray, 255)
                label += " (n/a)"
            cells.append(tile(ov, label))
        row = np.hstack([np.full((cells[0].shape[0], GAP, 3), 200, np.uint8)] +
                        [c for cell in cells for c in
                         (cell, np.full((cell.shape[0], GAP, 3), 200, np.uint8))])
        sheet = np.vstack([np.hstack(cells)])  # single sample sheet
        cv2.imwrite(str(out_dir / f"{img_path.stem}_compare.png"), sheet)
        all_rows.append(row)

    if all_rows:
        page = np.vstack(all_rows)  # one row per sample: easier to view
        cv2.imwrite(str(out_dir / "all_compare.png"), page)
        print("saved per-sample: " +
              ", ".join(str(out_dir / f"{p[1].stem}_compare.png") for p in pairs))
        print("saved combined  : " + str(out_dir / "all_compare.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
