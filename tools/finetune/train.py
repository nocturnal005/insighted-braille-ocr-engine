"""Fine-tune the DotNeuralNet braille-cell detector on reviewed local data.

Small-dataset settings: start from the pretrained braille weights, freeze the
backbone, low learning rate, photometric augmentation only (no mosaic/flips -
braille is orientation- and layout-sensitive; a vertical flip changes cell
identities).

Refuses to run with fewer than MIN_TRAIN_IMAGES reviewed training images:
below that a fine-tune memorises the samples and any resulting "accuracy" is
fiction. Collect more transcribed pages instead (see README.md).

Privacy: dataset and run outputs must be outside the repository. Weights
trained on pupil worksheets inherit that sensitivity.

Usage:
    python train.py --data "D:/braille training/dataset/data.yaml" \
        --weights "path/to/yolov8_braille.pt" \
        --out "D:/braille training/runs" [--epochs 80]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Below this, do not pretend to train. 4 pages is a diagnostic, not a dataset.
MIN_TRAIN_IMAGES = 30


def _refuse_repo_paths(*paths: Path) -> None:
    for path in paths:
        resolved = path.resolve()
        if resolved == _REPO_ROOT or _REPO_ROOT in resolved.parents:
            sys.exit(
                f"refusing: {path} is inside the engine repository. Training "
                "data and runs must live outside it (tools/finetune/README.md)."
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="data.yaml from prepare_dataset")
    parser.add_argument("--weights", required=True, help="yolov8_braille.pt path")
    parser.add_argument("--out", required=True, help="External runs directory")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument(
        "--allow-tiny-dataset", action="store_true",
        help="Override the minimum-images guard for smoke tests only. "
        "Never present the result as accuracy.",
    )
    args = parser.parse_args()

    data_yaml = Path(args.data)
    out_dir = Path(args.out)
    _refuse_repo_paths(data_yaml, out_dir)
    if not data_yaml.is_file():
        sys.exit(f"data.yaml not found: {data_yaml}")

    train_images_dir = data_yaml.parent / "images" / "train"
    n_train = len(list(train_images_dir.glob("*.jpg"))) if train_images_dir.is_dir() else 0
    if n_train < MIN_TRAIN_IMAGES and not args.allow_tiny_dataset:
        sys.exit(
            f"only {n_train} training image(s); need >= {MIN_TRAIN_IMAGES} "
            "reviewed pages for a fine-tune that means anything. Collect and "
            "transcribe more captures first (README.md), or pass "
            "--allow-tiny-dataset for a smoke test."
        )

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("pip install ultralytics (separate venv)")

    model = YOLO(args.weights)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=str(out_dir),
        name="braille_finetune",
        # Small-dataset regime: keep the pretrained backbone's features.
        freeze=10,
        lr0=0.001,
        optimizer="AdamW",
        # Photometric-only augmentation. Geometric flips are DISABLED on
        # purpose: braille cell classes are defined by dot position, so a
        # flip silently relabels every cell.
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.0,
        degrees=3.0,
        translate=0.05,
        scale=0.1,
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.4,
        patience=20,
    )
    print(
        "\nDone. Evaluate on held-out transcribed pages with the engine's "
        "WER/CER metrics before drawing any conclusion. The weights inherit "
        "the training data's privacy sensitivity."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
