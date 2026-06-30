# Photorealistic Animal Detection Pipeline (YOLOv8)

A fully self-contained computer vision pipeline that:
1. **Generates** a synthetic dataset of 3,000 photorealistic animal images using Stable Diffusion
2. **Trains** a YOLOv8s object detection model on that data
3. **Predicts** animal classes on new images with a live accuracy report
4. **Evaluates** formal detection metrics (mAP, Precision, Recall) against labelled ground truth

Supported classes: **Fox** · **Leopard** · **Sheep**

---

## Table of Contents

- [What's in This Repo](#whats-in-this-repo)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Directory Structure](#directory-structure)
- [Script Reference](#script-reference)
  - [generate_assets.py](#generate_assetspy)
  - [train.py](#trainpy)
  - [predict.py](#predictpy)
  - [evaluate.py](#evaluatepy)
  - [normalize_labels.py](#normalize_labelspy)
  - [voc_to_yolo.py](#voc_to_yolopy)
- [Pipeline A — Train from Scratch](#pipeline-a--train-from-scratch)
- [Pipeline B — Use the Included best.pt](#pipeline-b--use-the-included-bestpt)
- [Working with External Datasets](#working-with-external-datasets)
- [Troubleshooting](#troubleshooting)

---

## What's in This Repo

| File | Purpose |
|---|---|
| `generate_assets.py` | Generates the synthetic dataset using Stable Diffusion + RemBG |
| `train.py` | Trains YOLOv8s on the generated dataset |
| `predict.py` | Runs inference on a single image or a folder |
| `evaluate.py` | Computes mAP, Precision, and Recall against labelled ground truth |
| `normalize_labels.py` | Fixes unnormalised bounding box coordinates in external datasets |
| `voc_to_yolo.py` | Converts Pascal VOC XML annotations to YOLO `.txt` format |
| `requirements.txt` | All Python dependencies |
| `best.pt` | Pre-trained YOLOv8s weights (~22 MB) — ready to use immediately |

---

## Prerequisites

### Python

Python **3.10 or later** is required. The codebase uses `tuple[X, Y]` type hints which are only valid from 3.10+.

### GPU (strongly recommended)

| Component | Minimum | Recommended |
|---|---|---|
| CUDA GPU VRAM | 6 GB | 10 GB+ |
| System RAM | 12 GB | 16 GB+ |
| Free disk space | 15 GB | 30 GB |

`generate_assets.py` and `train.py` both fall back to CPU if no GPU is detected, but generation will be very slow on CPU (hours vs. minutes per class). Inference and evaluation are fast even on CPU.

### CUDA

Install the CUDA-compatible version of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) **before** running `pip install -r requirements.txt`. The requirements file will install a CPU-only PyTorch if you skip this step.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/ojasjog/animal-detector/
cd animal-detector

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install PyTorch with CUDA first (example for CUDA 12.1 — check pytorch.org for your version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install all other dependencies
pip install -r requirements.txt

# 5. Log in to HuggingFace (required to download the Stable Diffusion model)
huggingface-cli login
```

> **No GPU?** Replace `rembg[gpu]` with `rembg[cpu]` in `requirements.txt` before running step 4.

> **HuggingFace token:** Create a free account at [huggingface.co](https://huggingface.co) and get your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). The model used does not require any special access agreement.

---

## Directory Structure

The layout below shows the full project after running `generate_assets.py` and `train.py`. Files marked with ★ are included in the repo before you run anything.

```
animal-detector/
│
├── best.pt ★                      ← Pre-trained weights (included, ~22 MB)
├── generate_assets.py ★
├── train.py ★
├── predict.py ★
├── evaluate.py ★
├── normalize_labels.py ★
├── voc_to_yolo.py ★
├── requirements.txt ★
│
├── combined_dataset/              ← Created by generate_assets.py
│   ├── data.yaml                  ← Auto-generated YOLO dataset config
│   ├── images/
│   │   ├── train/                 ← 2,400 composite images (800 per class)
│   │   └── val/                   ← 600 composite images (200 per class)
│   └── labels/
│       ├── train/                 ← 2,400 YOLO .txt label files
│       └── val/                   ← 600 YOLO .txt label files
│
└── yolo_backups/                  ← Created by train.py
    └── animal_detector_v8s_run/
        ├── weights/
        │   ├── best.pt            ← Best epoch by validation mAP
        │   └── last.pt            ← Final epoch
        ├── results.csv            ← Per-epoch loss and metric history
        └── ...                    ← Confusion matrix, PR curves, plots
```

---

## Script Reference

---

### `generate_assets.py`

Generates the full synthetic training dataset. Uses Stable Diffusion to create photorealistic animals, removes their backgrounds with alpha matting, composites them onto generated landscape scenes, and writes pixel-precise YOLO bounding box labels for every image.

**No CLI arguments.** Edit the constants at the top of the file to change any behaviour.

#### Configuration Constants

| Constant | Default | Description |
|---|---|---|
| `MODEL_ID` | `"SG161222/Realistic_Vision_V5.1_noVAE"` | HuggingFace repo ID for the Stable Diffusion checkpoint |
| `VAE_ID` | `"stabilityai/sd-vae-ft-mse"` | HuggingFace repo ID for the fine-tuned VAE |
| `CLASSES` | `{0: "fox", 1: "leopard", 2: "sheep"}` | Class index-to-name map; must match `data.yaml` |
| `CANVAS_W / CANVAS_H` | `640, 640` | Output image resolution; keep at 640 to match YOLO's native grid |
| `FG_SCALE_MIN / FG_SCALE_MAX` | `0.25, 0.55` | Min/max animal size as a fraction of canvas width |
| `NUM_AUGMENTED_IMAGES` | `1000` | Images to generate per class (800 go to train, 200 go to val) |
| `SEED` | `42` | Global random seed for reproducible augmentation |
| `OUTPUT_DIR` | `Path("combined_dataset")` | Root output directory |
| `BASE_SIZE` | `(512, 512)` | Resolution for the initial txt2img generation pass |
| `HIRES_SIZE` | `(640, 640)` | Resolution after LANCZOS upscale and img2img refinement |
| `HIRES_DENOISE` | `0.45` | img2img denoising strength during the hires fix pass |
| `BASE_STEPS` | `18` | Inference steps for the base txt2img pass |
| `HIRES_STEPS` | `12` | Inference steps for the hires img2img refinement pass |
| `CFG_SCALE` | `5.5` | Classifier-free guidance scale for animal foreground generation |
| `BG_CFG_SCALE` | `5.5` | Classifier-free guidance scale for background generation |

#### Models Used

| Model | Identifier | Role |
|---|---|---|
| Stable Diffusion | `SG161222/Realistic_Vision_V5.1_noVAE` | Photorealistic image generation (txt2img + img2img passes) |
| Fine-tuned VAE | `stabilityai/sd-vae-ft-mse` | Improved colour fidelity; injected into both SD pipelines |
| Background remover | `isnet-general-use` via `rembg` | Alpha-matting-based foreground isolation |

Both SD pipelines (txt2img and img2img) share the same `text_encoder`, `tokenizer`, and `unet` to avoid loading the model twice into VRAM.

#### Device and Memory Handling

The script auto-detects the available hardware at startup:

- **CUDA available** → models loaded in `float16`, run on GPU
- **No CUDA** → models loaded in `float32`, run on CPU (very slow)

Memory optimisations are applied to each pipeline in this priority order:

1. **xformers** memory-efficient attention — fastest; install with `pip install xformers`
2. **PyTorch 2.0 `AttnProcessor2_0`** — uses built-in `scaled_dot_product_attention`; no extra install needed on torch ≥ 2.0
3. **`enable_sequential_cpu_offload()`** — slowest and lowest VRAM usage; used as a last resort

#### What the Script Does, Step by Step

**Phase 1 — Cache backgrounds** *(runs once, 10 images total)*

Ten distinct landscape scenes are generated with the same hires pipeline as the animals, then held in memory for reuse across all composites. A shared negative prompt explicitly removes people, animals, birds, insects, text, and CGI artefacts from every background.

| # | Scene Description |
|---|---|
| 1 | Sunny forest clearing |
| 2 | Grassy meadow at golden hour |
| 3 | Rocky hillside with sparse dry trees |
| 4 | Snowy woodland path |
| 5 | Autumn forest floor with fallen leaves |
| 6 | Riverbank with pebbles and wild grass |
| 7 | Wide countryside field at dusk |
| 8 | Sunny open savanna |
| 9 | Dense jungle interior |
| 10 | Dry scrubland desert |

**Phase 2 — Generate animal foregrounds** *(15 templates per class)*

For each class, 15 distinct prompt templates cover a range of poses and camera angles: sitting, standing alert, lying down, walking (side profile), running mid-stride, crouching, stretching, curling up, head-tilt portrait, looking over shoulder, low-angle, macro close-up, yawning, three-quarter portrait, and stepping forward. A shared negative prompt suppresses blur, watermarks, cartoons, multiple animals, and anatomical errors.

Each foreground image goes through a two-pass hires pipeline:

1. **Base pass** — 512×512, DPM++ 2M Karras scheduler, 18 inference steps, CFG 5.5
2. **Hires fix** — upscale to 640×640 with LANCZOS, then img2img at denoising strength 0.45, 12 steps

After generation, `rembg` removes the background using alpha matting (`foreground_threshold=240`, `background_threshold=10`, `erode_size=10`), and the result is cropped to its content bounding box to remove transparent padding.

**Phase 3 — Composite and label** *(1,000 images per class)*

For each image in the batch:

1. A foreground template is selected cyclically (`i % 15`).
2. Augmentations are applied to the RGBA foreground. The RGB and alpha channels are augmented **separately with the same seeded random state** so they remain pixel-aligned:

   | Transform | Settings | Probability |
   |---|---|---|
   | `HorizontalFlip` | — | 0.5 |
   | `RandomBrightnessContrast` | ±35% brightness, ±35% contrast | 0.8 |
   | `ColorJitter` | brightness 0.2, contrast 0.2, saturation 0.2, hue 0.1 | 0.6 |
   | `CoarseDropout` | up to 12 holes, max 32×32 px, filled with 0 | 0.4 |
   | `MotionBlur` | blur kernel limit 5 | 0.3 |
   | `Rotate` | ±15°, border fill 0 | 0.5 |

3. The foreground is randomly scaled to **25–55% of canvas width** (aspect ratio preserved).
4. Pasted at a random `(x, y)` offset onto a randomly selected cached background.
5. The YOLO label is derived directly from the **foreground alpha channel** (pixel threshold 10) — not the outer tile bounding box — giving pixel-precise labels. Coordinates are normalised to [0, 1].

**Split:** images 0–799 per class → `train/`; images 800–999 → `val/`

**Output naming:** `sample_{class_id}_{index:04d}.jpg` and `sample_{class_id}_{index:04d}.txt`

**`data.yaml`** is written automatically with the absolute resolved path of `combined_dataset/`, `train: images/train`, `val: images/val`, `nc: 3`, and `names: ['fox', 'leopard', 'sheep']`.

---

### `train.py`

Trains a YOLOv8s model on `combined_dataset/` with domain-shift defences designed to help a model trained on synthetic images generalise to real-world photos.

**No CLI arguments.** Edit values directly in the file.

The script exits immediately with a clear error if `combined_dataset/data.yaml` does not exist, reminding you to run `generate_assets.py` first.

#### Configuration

| Parameter | Value | Notes |
|---|---|---|
| Base weights | `yolov8s.pt` | Small YOLOv8 variant; downloaded automatically by Ultralytics on first run |
| `epochs` | `60` | Total training epochs |
| `imgsz` | `640` | Input image size; matches the canvas size from `generate_assets.py` |
| `batch` | `16` | Batch size per gradient step; reduce to `8` or `4` if you get out-of-memory errors |
| `device` | `0` | GPU device index; change to `'cpu'` if no CUDA GPU is available |
| `workers` | `2` | DataLoader subprocess count |
| `project` | `"yolo_backups"` | Parent directory for all run outputs |
| `name` | `"animal_detector_v8s_run"` | Subdirectory name under `project` |
| `save_period` | `15` | Save an intermediate checkpoint every 15 epochs |

#### Domain-Shift Defences

These augmentations run during training to help the model bridge the gap between synthetic training images and real-world test photos.

| Parameter | Value | What It Does |
|---|---|---|
| `mosaic` | `1.0` | Stitches 4 training images into one composite frame, forcing the model to handle occluded and partially visible animals |
| `close_mosaic` | `10` | Disables mosaic in the **final 10 epochs** so gradients converge cleanly on unmodified images |
| `mixup` | `0.20` | Blends two training images with a transparency mask; improves class boundary robustness |
| `scale` | `0.6` | Random scale jitter; simulates animals at different distances from the camera |
| `perspective` | `0.0005` | Random camera warp; normalises for lens distortion |

---

### `predict.py`

Runs inference using a trained model. Accepts a single image or an entire folder. In folder mode, every prediction is compared against the **folder name** and a full accuracy report is printed at the end.

#### CLI Arguments

`--image` and `--folder` are mutually exclusive. Exactly one must be provided.

| Flag | Type | Default | Required | Description |
|---|---|---|---|---|
| `--image` | `str` | — | one of these two | Path to a single image file |
| `--folder` | `str` | — | one of these two | Path to a folder of images |
| `--weights` | `str` | `best.pt` | No | Path to model weights |
| `--conf` | `float` | `0.25` | No | Confidence threshold (0–1); detections below this are discarded |
| `--iou` | `float` | `0.45` | No | IoU threshold for Non-Maximum Suppression |
| `--save` | flag | off | No | Saves an annotated copy of each image as `{original_stem}_predicted{ext}` next to the original |

#### Supported Image Formats

`.jpg` · `.jpeg` · `.png` · `.bmp` · `.webp` · `.tiff`

#### How Predictions Work

The model's output boxes are filtered by `--conf`. If any boxes pass, the box with the **highest confidence score** (`boxes.conf.argmax()`) is used as the single prediction for that image. The predicted class name is read from `r.names[best_cls]`.

Ground truth is inferred from the **immediate parent folder name**, lowercased. An image at `Fox/img001.jpg` has a ground truth of `fox`. The comparison is case-insensitive. Images with no detection above the confidence threshold count as incorrect in the accuracy calculation.

#### Example Output — Single Image

```
[INFO] Loading model from: best.pt

  fox_001.jpg                              →  This is a fox image  (conf: 94.3%)  [✓ MATCH with 'fox']

This is a fox image  (conf: 94.3%)
```

#### Example Output — Folder

```
[INFO] Found 5 image(s) in Fox

  File                                         Prediction
  ----------------------------------------     -------------------------------------------------------
  fox_001.jpg                              →  This is a fox image  (conf: 94.3%)  [✓ MATCH with 'fox']
  fox_002.jpg                              →  This is a leopard image  (conf: 61.1%)  [✗ MISMATCH with 'fox']
  fox_003.jpg                              →  no detection (conf≥25%) [GT: fox]
  fox_004.jpg                              →  This is a fox image  (conf: 88.7%)  [✓ MATCH with 'fox']
  fox_005.jpg                              →  This is a fox image  (conf: 79.2%)  [✓ MATCH with 'fox']

── Summary ────────────────────────────────────────────────────────
  Target Folder  : Fox
  Total images   : 5
  Detected       : 4
  No detection   : 1
  Correct Match  : 3 / 5
  Folder Accuracy: 60.00%
  Breakdown      :
    fox                  3 image(s)
    leopard              1 image(s)
──────────────────────────────────────────────────────────────────
```

---

### `evaluate.py`

Computes formal object detection metrics against YOLO-format ground-truth `.txt` label files. Use this when you need standardised mAP numbers rather than the folder-name accuracy check in `predict.py`.

**Requires** a dataset with YOLO `.txt` label files alongside the images, and a `data.yaml` pointing to them.

#### CLI Arguments

| Flag | Type | Default | Required | Description |
|---|---|---|---|---|
| `--weights` | `str` | `best.pt` | No | Path to model weights |
| `--data` | `str` | — | **Yes** | Path to a YOLO-format dataset YAML |
| `--split` | `str` | `val` | No | Dataset split to evaluate: `train`, `val`, or `test` |
| `--imgsz` | `int` | `640` | No | Inference image size |
| `--device` | `str` | `0` | No | `0` for GPU, `'cpu'` for CPU |

#### Metrics Reported

| Metric | What It Measures |
|---|---|
| `mAP@0.50` | Mean Average Precision at IoU ≥ 0.50 — the standard Pascal VOC metric |
| `mAP@0.50:0.95` | mAP averaged over IoU thresholds 0.50–0.95 in steps of 0.05 — the stricter COCO standard |
| `Precision (P)` | Of all predicted boxes, what fraction are correct |
| `Recall (R)` | Of all ground-truth objects, what fraction were detected |

The script exits with a clear error if the weights file is not found.

#### Example Usage

```bash
# Evaluate on the val split of the generated dataset
python evaluate.py --weights best.pt --data combined_dataset/data.yaml

# Evaluate on a custom dataset's test split, on CPU
python evaluate.py --weights best.pt --data path/to/my_data.yaml --split test --device cpu
```

---

### `normalize_labels.py`

Scans `combined_dataset/labels/train/` and `combined_dataset/labels/val/` and converts any YOLO label files that contain absolute pixel coordinates into the normalised [0, 1] format that Ultralytics requires. Safe to run on an already-correct dataset — files that are already normalised are left untouched.

**No CLI arguments.** The dataset root is a hardcoded constant.

#### Configuration Constant

| Constant | Default | Description |
|---|---|---|
| `DATASET_ROOT` | `Path("combined_dataset")` | Root of the dataset to scan. Labels are read from `DATASET_ROOT/labels/{train,val}/` and matching images are found in `DATASET_ROOT/images/{train,val}/` |

#### When to Run This

Run `normalize_labels.py` **before** `train.py` or `evaluate.py` whenever you are using an externally sourced dataset. Absolute pixel coordinates in label files cause training crashes or silently wrong mAP scores.

#### How It Works

For every `.txt` file in both the train and val splits:

1. Searches for a matching image by trying extensions `.jpg`, `.jpeg`, `.png`, `.JPG` in that order.
2. Opens the matching image with PIL to read the true pixel width and height.
3. Reads each label line and checks whether **any of the four coordinate values exceeds 1.0**.
4. If unnormalised values are found, interprets them as `xmin ymin xmax ymax` in absolute pixels, converts to YOLO centre-format `cx cy w h`, divides by image dimensions, and clips all values to [0.0, 1.0].
5. Overwrites the file only if at least one line was actually changed.

Lines with fewer than 5 fields are silently skipped. Individual file errors are printed and skipped without stopping the batch. A per-split count of fixed files is printed at the end.

---

### `voc_to_yolo.py`

Converts Pascal VOC `.xml` annotation files to YOLO `.txt` label files. Run this before `normalize_labels.py` if your external dataset ships with XML annotations.

**No CLI arguments.** All paths and class mappings are hardcoded constants.

#### Configuration Constants

| Constant | Default | Description |
|---|---|---|
| `XML_DIR` | `Path("annotations/")` | Folder containing the source `.xml` files |
| `TXT_OUT_DIR` | `Path("labels/")` | Output folder for the converted `.txt` files; created automatically if it does not exist |
| `CLASS_MAPPING` | `{"fox": 0, "leopard": 1, "sheep": 2}` | Maps the class name string from `<name>` XML tags to a YOLO integer ID. Must match the `names` order in your `data.yaml`. Keys must be **lowercase**. |

> **Important:** Output goes to a flat `labels/` folder, **not** into `combined_dataset/labels/`. After conversion, move the `.txt` files into `combined_dataset/labels/train/` or `combined_dataset/labels/val/` as appropriate before training.

#### Conversion Logic

For each `.xml` file in `XML_DIR`:

1. Reads `<size>/<width>` and `<size>/<height>` for the image dimensions.
2. Iterates over all `<object>` elements. Any class name not present in `CLASS_MAPPING` is silently skipped.
3. Reads `<bndbox>` coordinates (`xmin`, `ymin`, `xmax`, `ymax`) as absolute pixel values.
4. Converts to YOLO normalised centre-format:
   ```
   cx     = (xmin + (xmax - xmin) / 2) / img_width
   cy     = (ymin + (ymax - ymin) / 2) / img_height
   norm_w = (xmax - xmin) / img_width
   norm_h = (ymax - ymin) / img_height
   ```
5. Writes a `.txt` file to `TXT_OUT_DIR` with the same name stem as the source XML. Files that fail to parse are reported individually and skipped. A final total of successfully converted files is printed.

---

## Pipeline A — Train from Scratch

Follow these steps if you want to generate fresh synthetic data and train a new model.

### Step 1 — Generate the dataset

```bash
python generate_assets.py
```

On first run this downloads `Realistic_Vision_V5.1_noVAE` and `sd-vae-ft-mse` from HuggingFace (~4 GB total, cached locally after the first run). It then generates 10 background landscapes, 15 foreground templates per class, and composites 3,000 labelled training images.

**Expected time on a 10 GB VRAM GPU:** ~20–40 minutes.

### Step 2 — Train the model

```bash
python train.py
```

Trains for 60 epochs with intermediate checkpoints every 15 epochs. The best checkpoint (by validation mAP) is saved to `yolo_backups/animal_detector_v8s_run/weights/best.pt`.

**Expected time on a modern GPU:** ~30–60 minutes.

### Step 3 — Predict on new images

```bash
# Single image
python predict.py --image path/to/image.jpg

# Entire folder — folder name must match the animal class for accuracy reporting
python predict.py --folder path/to/Fox/

# Save annotated output images alongside originals
python predict.py --folder path/to/Fox/ --save

# Raise the confidence bar to reduce false positives
python predict.py --folder path/to/Fox/ --conf 0.5
```

### Step 4 — Formal evaluation (optional)

```bash
python evaluate.py \
  --weights yolo_backups/animal_detector_v8s_run/weights/best.pt \
  --data combined_dataset/data.yaml
```

---

## Pipeline B — Use the Included `best.pt`

A pre-trained `best.pt` (~22 MB) is included in this repository. You can skip generation and training entirely.

### Run inference immediately

```bash
# Single image (best.pt is the default weight, so --weights is optional)
python predict.py --image path/to/image.jpg

# Folder with live accuracy report
python predict.py --folder path/to/Leopard/
```

### Evaluate against your own labelled dataset

```bash
python evaluate.py --weights best.pt --data path/to/your_data.yaml
```

Your `data.yaml` must use this exact class ordering:

```yaml
nc: 3
names: ['fox', 'leopard', 'sheep']
```

---

## Working with External Datasets

### Your dataset has Pascal VOC XML annotations

```bash
# 1. Place your .xml annotation files in annotations/
#    Edit CLASS_MAPPING in voc_to_yolo.py if your class names differ

python voc_to_yolo.py

# 2. Move the output .txt files into the right split folder
mv labels/*.txt combined_dataset/labels/train/   # adjust to val/ as needed

# 3. Fix any unnormalised coordinates
python normalize_labels.py
```

### Your dataset has YOLO .txt labels but may have unnormalised coordinates

```bash
python normalize_labels.py
```

This is safe to run even on already-normalised data.

### Adding more animal classes

1. Add entries to `CLASSES` in `generate_assets.py` and write corresponding prompts in `ANIMAL_PROMPTS`.
2. Update `CLASS_MAPPING` in `voc_to_yolo.py` if you are also converting XML data.
3. Re-run `generate_assets.py` — `data.yaml` will be rewritten automatically with the updated class list.
4. Re-run `train.py`.

---

## Troubleshooting

**`[ERROR] Configuration profile not found at combined_dataset/data.yaml`**
Run `generate_assets.py` first, or place your dataset under `combined_dataset/` with a valid `data.yaml`.

**`[ERROR] Model weights not found: best.pt`**
Either run `train.py` to produce weights, or point to the included file explicitly with `--weights best.pt`.

**`[ERROR] No images found in <folder>`**
The folder must contain files with supported extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tiff`.

**`CUDA out of memory` during generation**
The script automatically tries the memory optimisation fallback chain (xformers → AttnProcessor2_0 → sequential CPU offload). If it still fails, reduce `BASE_SIZE` to `(448, 448)` in `generate_assets.py`.

**`CUDA out of memory` during training**
Reduce `batch` in `train.py` — try `8` or `4`. You can also reduce `imgsz` to `512`, though this will slightly reduce accuracy since it no longer matches the generation canvas size.

**Bounding box out-of-range warnings during training**
Run `python normalize_labels.py` before training. This means your label files contain absolute pixel coordinates instead of normalised values.

**Generation is very slow**
Check that PyTorch can see your GPU:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
If it prints `False`, reinstall PyTorch with the correct CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).

**HuggingFace download fails**
Run `huggingface-cli login` and enter your access token. The model `Realistic_Vision_V5.1_noVAE` does not require any gated access agreement.

**`xformers` not found warning**
Non-fatal — the script falls back automatically to PyTorch 2.0 attention or CPU offload. To silence it and get a speed boost: `pip install xformers`.

**Folder accuracy is 0% even though detections are being made**
The accuracy check compares the predicted class name against the folder name, lowercased. Make sure your test images are inside a folder named exactly `fox`, `leopard`, or `sheep`. A folder named `foxes` or `red_fox` will not match.
