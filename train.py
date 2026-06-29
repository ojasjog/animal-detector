"""
YOLOv8s Domain Blended Training Script
=======================================
Trains a YOLOv8s model on the combined synthetic + real animal dataset.
Saves checkpoints to a local 'yolo_backups' directory by default.
"""

import sys
from pathlib import Path
from ultralytics import YOLO


def main():
    # 1. Backup directory
    backup_root = Path("yolo_backups")
    backup_root.mkdir(parents=True, exist_ok=True)

    dataset_yaml = Path("combined_dataset/data.yaml")
    if not dataset_yaml.exists():
        print(f"[ERROR] Configuration profile not found at {dataset_yaml}")
        print("        Run generate_assets.py first, or place your dataset under combined_dataset/")
        sys.exit(1)

    print("\n============================================================")
    print("  YOLOv8s Blended Domain Training Routine Initialized")
    print("============================================================\n")

    # 2. Load YOLOv8s base weights
    model = YOLO("yolov8s.pt")

    # 3. Fire training with defensive augmentation profiles
    results = model.train(
        data=str(dataset_yaml),
        epochs=60,                          # The sweet spot window
        imgsz=640,                          # Matches native canvas dimensions
        batch=16,                           # Stable gradient step size
        device=0,                           # GPU (change to 'cpu' if no GPU)
        workers=2,

        # ── DOMAIN SHIFT DEFENSES ────────────────────────────────────────────
        mosaic=1.0,                         # Multi-image stitching
        close_mosaic=10,                    # Drop spatial distortion in final 10 epochs
        mixup=0.20,                         # Transparent entity blending
        scale=0.6,                          # Simulates distance scaling
        perspective=0.0005,                 # Camera lens warp normalization

        # ── CHECKPOINT BACKUPS ────────────────────────────────────────────────
        project=str(backup_root),
        name="animal_detector_v8s_run",
        save=True,
        save_period=15,                     # Checkpoint every 15 epochs
    )

    print("\n✅ Training pipeline successfully finished.")
    print(f"[INFO] Weights saved to: {backup_root}/animal_detector_v8s_run/weights")


if __name__ == "__main__":
    main()
