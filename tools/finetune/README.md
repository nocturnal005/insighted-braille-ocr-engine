# Braille cell detector fine-tuning workstream

## Why this exists

The 2026-07-12 pretrained-model spike showed the accuracy problem splits in
two on real worksheet photos:

- **Localisation** (where are the cells / lines): the pretrained
  [DotNeuralNet](https://github.com/snoop2head/DotNeuralNet) YOLOv8 braille
  detector (MIT) already does this well — ~300 cells / correct line counts on
  pages where our geometric pipeline finds nothing.
- **Classification** (which 6-dot pattern each cell is): both our pipeline
  and the pretrained model are ~0% here. Faint white-on-white embossing on
  our specific paper is out-of-domain for every off-the-shelf model tested.

Classification on in-domain captures is exactly what fine-tuning fixes, and
the raw material now exists: real worksheet photos plus word-level teacher
transcriptions.

## The honest blocker: data volume

Four transcribed pages is a diagnostic sample, not a training set. A model
fine-tuned on 4 pages will memorise them. Before training means anything:

- **Target: 50+ transcribed pages minimum** (hundreds preferable), captured
  under the guided-capture conditions (plain background, page fills frame,
  upright, good light — see Stage 3D-L1).
- Every page needs a permission decision: these are pupil worksheets. The
  Stage 3D-G6 protocol (permission metadata, safe naming) applies to any
  captures promoted into formal evaluation, and the same standard should be
  met before pupil work is used as training data.

## Privacy rules (non-negotiable)

- No images, labels, weights, or dataset manifests are ever committed to
  this repo. The scripts refuse to write inside the repository and default
  to external directories.
- Training happens locally. Nothing is uploaded to any service.
- Fine-tuned weights inherit the training data's sensitivity: treat them as
  containing pupil data unless a review says otherwise.

## Workflow

```
1. Collect photos (guided capture) + teacher transcriptions
        |
2. prepare_dataset.py  — pretrained DotNeuralNet weights pseudo-label the
        |                cell boxes; output is a YOLO-format dataset in an
        |                EXTERNAL folder, one label file per image
3. Human review        — correct the pseudo-labels (box + 6-dot class) in
        |                any YOLO-compatible annotator (e.g. Label Studio,
        |                makesense.ai used offline). The transcription tells
        |                the reviewer what each cell SHOULD be.
4. train.py            — fine-tune from yolov8_braille.pt on the corrected
        |                dataset (small-dataset settings: frozen backbone,
        |                low LR, photometric augmentation)
5. Evaluate            — score on held-out transcribed pages with the
                         engine's own WER/CER metrics before believing
                         anything
```

## Setup (separate environment — NOT the engine venv)

```
python -m venv .venv-finetune
.venv-finetune/Scripts/pip install ultralytics opencv-python-headless
```

Weights: clone DotNeuralNet and use `weights/yolov8_braille.pt` (MIT licence;
keep the attribution). The engine itself does not depend on ultralytics —
this workstream is offline tooling only until a trained model earns its way
in through measured accuracy.
