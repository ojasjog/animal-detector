"""
Model Evaluation Script
========================
Runs validation on a trained YOLOv8 model and prints a performance report.

Usage:
    python evaluate.py --weights best.pt --data path/to/test_data.yaml
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Evaluate a YOLOv8 model")
    parser.add_argument("--weights", type=str, default="best.pt", help="Path to model weights (.pt)")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset YAML (e.g. test_data.yaml)")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to evaluate on")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="0", help="Device: 0 for GPU, 'cpu' for CPU")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[ERROR] Weights not found at {weights_path}")
        return

    model = YOLO(str(weights_path))

    print("\nExecuting Validation Loop...\n")

    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
        verbose=True
    )

    print("\n============================================================")
    print("  PERFORMANCE REPORT")
    print("============================================================")
    print(f"  mAP@0.50 (mAP50)       : {metrics.box.map50:.4f}")
    print(f"  mAP@0.50:0.95 (mAP50-95): {metrics.box.map:.4f}")
    print(f"  Precision (P)          : {metrics.box.mp:.4f}")
    print(f"  Recall (R)             : {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
