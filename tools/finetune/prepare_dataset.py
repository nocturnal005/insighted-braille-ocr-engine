"""Pseudo-label real Braille captures into a reviewable YOLO dataset.

Runs the pretrained DotNeuralNet YOLOv8 braille-cell detector over a folder
of captures and writes a YOLO-format dataset (images + label txts + data.yaml)
to an EXTERNAL directory for human correction. Pseudo-labels are a starting
point for review, never training truth: every box and class must be checked
against the page's transcription before train.py is run.

Privacy: refuses to write inside the repository. No image content is logged.

Usage:
    python prepare_dataset.py --images "D:/braille training/raw" \
        --weights "path/to/yolov8_braille.pt" \
        --out "D:/braille training/dataset" [--val-fraction 0.2]
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# Same normalisation the engine applies (Stage 3D-L1): the detector should
# be fine-tuned on what the pipeline will actually feed it.
_MAX_LONG_SIDE = 1600
_TARGET_LONG_SIDE = 1400


def _refuse_repo_paths(*paths: Path) -> None:
    for path in paths:
        resolved = path.resolve()
        if resolved == _REPO_ROOT or _REPO_ROOT in resolved.parents:
            sys.exit(
                f"refusing: {path} is inside the engine repository. Training "
                "data must live in an external folder (privacy rule - see "
                "tools/finetune/README.md)."
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="Folder of captures")
    parser.add_argument("--weights", required=True, help="yolov8_braille.pt path")
    parser.add_argument("--out", required=True, help="External output dataset dir")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    images_dir = Path(args.images)
    out_dir = Path(args.out)
    _refuse_repo_paths(images_dir, out_dir)

    try:
        import cv2
        from ultralytics import YOLO
    except ImportError:
        sys.exit("pip install ultralytics opencv-python-headless (separate venv)")

    candidates = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in _IMAGE_SUFFIXES and p.is_file()
    )
    if not candidates:
        sys.exit(f"no images found in {images_dir}")

    model = YOLO(args.weights)
    rng = random.Random(args.seed)
    split_of = {
        p: ("val" if rng.random() < args.val_fraction else "train")
        for p in candidates
    }
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    labelled = 0
    for path in candidates:
        image = cv2.imread(str(path))
        if image is None:
            print(f"skip (unreadable): {path.name}")
            continue
        h, w = image.shape[:2]
        long_side = max(h, w)
        if long_side > _MAX_LONG_SIDE:
            scale = _TARGET_LONG_SIDE / long_side
            image = cv2.resize(
                image, (round(w * scale), round(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        result = model.predict(
            image, conf=args.conf, imgsz=1280, max_det=2000, verbose=False
        )[0]
        split = split_of[path]
        stem = path.stem.replace(" ", "_")
        cv2.imwrite(str(out_dir / "images" / split / f"{stem}.jpg"), image)
        lines = []
        if result.boxes is not None:
            for xywhn, cls in zip(
                result.boxes.xywhn.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
            ):
                cx, cy, bw, bh = (float(v) for v in xywhn)
                lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (out_dir / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        labelled += 1
        print(f"{stem}: {len(lines)} pseudo-boxes ({split})")

    names = model.names
    yaml_lines = [
        f"path: {out_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ] + [f"  {i}: {names[i]}" for i in sorted(names)]
    (out_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    print(
        f"\n{labelled} image(s) pseudo-labelled into {out_dir}.\n"
        "NEXT: human-review every label against the page transcription "
        "before running train.py - pseudo-labels are a starting point, "
        "not truth."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
