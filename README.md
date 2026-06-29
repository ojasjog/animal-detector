# Photorealistic Animal Detection Pipeline (YOLOv8)

This repository contains an end-to-end computer vision pipeline for generating a synthetic dataset of photorealistic animals (Fox, Leopard, Sheep) using Diffusion models, training a YOLOv8 object detection model, and evaluating/predicting on real-world data.

---

## Project Structure and Component Breakdown

This project is modular by design. Here is exactly what each script does:

* **`generate_assets.py`**: The synthetic data engine. It uses Stable Diffusion and RemBG to generate photorealistic animals, isolates them using alpha matting, and composites them onto generated backgrounds. It applies heavy Kaggle-style augmentations and automatically creates perfectly accurate YOLO bounding boxes. Automatically splits data 80/20 into `train` and `val` directories.
* **`train.py`**: The model training loop. Loads YOLOv8s, applies domain-shift defenses (mosaic, mixup, perspective scaling), and trains the model on the `combined_dataset`. Saves regular backups to `yolo_backups/`.
* **`predict.py`**: The inference engine. Takes a pre-trained `best.pt` model and runs predictions on a single `--image` or an entire `--folder`. If testing a folder, it automatically checks the folder's name (e.g., `/Fox/`) against its predictions to output a live accuracy report.
* **`evaluate.py`**: The formal validation script. Computes standardized object detection metrics (mAP50, mAP50-95, Precision, Recall) using strict ground-truth YOLO labels (`.txt` files) on a specified dataset split.
* **`normalize_labels.py`**: A critical data-cleaning utility. Scans dataset label files and repairs formatting errors. It maps string literals (e.g., `"Fox"`) to YOLO integer IDs (`0`), and mathematically converts absolute pixel coordinates into YOLO's required `[0.0 - 1.0]` normalized percentages.
* **`voc_to_yolo.py`**: A conversion utility. If you download an external dataset that uses Pascal VOC XML annotations, this script extracts the XML data and converts it into the `.txt` format YOLO requires.
* **`requirements.txt`**: The dependency manifest. Ensures identical environments for PyTorch, Diffusers, Ultralytics, and Augmentation libraries.

---

## Pipeline 1: Self-Sustaining (Train from Scratch)

If you are generating the synthetic dataset from scratch and training a new model, follow these steps in order.

### 1. Generate the Synthetic Dataset

Run the generation script to create 3,000 synthetic composited images (1,000 per class).

```bash
python generate_assets.py

```

This will output the standard YOLO directory structure with an 80/20 Training/Validation split:

```text
combined_dataset/
├── data.yaml
├── images/
│   ├── train/ (2,400 images)
│   └── val/   (600 images)
└── labels/
    ├── train/ (2,400 text files)
    └── val/   (600 text files)

```

### 2. Train the Model

Train the YOLOv8 model on the newly generated synthetic images. Checkpoints and the final model weights will be saved to `yolo_backups/`.

```bash
python train.py

```

### 3. Run Inference (Prediction)

Use the trained model to predict animal classes and draw bounding boxes on new, unseen images.

```bash
python predict.py --folder <path_to_images> --save

```

---

## Pipeline 2: Using a Pre-Trained Model

If you already have a pre-trained `best.pt` file and want to run inference or evaluate an *existing* external dataset, follow these guidelines.

### 1. Basic Inference and Classification Check

You can skip generation and training entirely. Point the predict script at your image folder using the pre-trained weights.

```bash
python predict.py --weights best.pt --folder <path_to_images>

```

*Note: If you are doing a classification check without YOLO ground-truth label files, ensure your test images are inside a folder named exactly after the target animal (e.g., `/Fox/`). The script will use the folder name to generate a basic win/loss classification accuracy report.*

### 2. Standard Object Detection Evaluation (mAP)

To get an accurate Mean Average Precision (mAP) report, your existing dataset must have YOLO-formatted ground-truth `.txt` labels.

**Command:**

```bash
python evaluate.py --weights best.pt --data path/to/test_data.yaml

```

---

## CRITICAL: External Data Normalization and Formatting

The Ultralytics YOLOv8 library strictly requires bounding box labels to use **Integer Class IDs** (0, 1, 2) and **Normalized Coordinates** (between `0.0` and `1.0`). If you are using an existing dataset scraped from the internet, it likely contains unnormalized absolute pixel values (e.g., `450 300 200 100`) or string literals (e.g., `"Fox"` instead of `0`).

**Feeding string literals or unnormalized pixels directly into YOLO will crash the training or evaluation scripts.**

To safely align an existing dataset with YOLO's strict requirements, you must run the normalizer utility **before** training or evaluating:

```bash
python normalize_labels.py

```

**What `normalize_labels.py` does automatically:**

1. Scans your dataset's `labels/` directory for any `.txt` files.
2. Checks for String Literals: If it finds `"Fox"`, `"Leopard"`, or `"Sheep"`, it seamlessly maps them to `0`, `1`, and `2`.
3. Checks for Pixel Values: If it finds coordinate values greater than `1.0`, it opens the corresponding image, calculates the real width/height, and mathematically converts the absolute pixels into YOLO's required relative percentages.
4. Safely ignores files that are already correctly normalized to prevent "double-shrinking" your bounding boxes.

### XML Data Conversions

If your external dataset uses Pascal VOC (`.xml`) files instead of YOLO text files, run the XML converter first:

```bash
python voc_to_yolo.py

```