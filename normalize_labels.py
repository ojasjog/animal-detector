"""
Bounding Box Normalization Utility
====================================
Scans combined_dataset/labels/{train,val}/ for YOLO label files
that contain absolute pixel coordinates and converts them to the
normalized [0, 1] format that YOLO expects.

Run this if training throws bounding-box out-of-range warnings.
"""

import os
from pathlib import Path
from PIL import Image

# ─────────────────────────────── CONFIG ──────────────────────────────────────
DATASET_ROOT = Path("combined_dataset")


def normalize_all_labels():
    print("============================================================")
    print("  Executing Bounding Box Normalization Pipeline")
    print("============================================================\n")

    for split in ["train", "val"]:
        img_dir = DATASET_ROOT / "images" / split
        lbl_dir = DATASET_ROOT / "labels" / split

        if not img_dir.exists() or not lbl_dir.exists():
            continue

        print(f"[PROCESSING] Normalizing split data path: '{split}'...")
        fixed_count = 0

        for txt_path in lbl_dir.glob("*.txt"):
            # Find the matching image file
            img_path = None
            for ext in [".jpg", ".jpeg", ".png", ".JPG"]:
                possible_path = img_dir / f"{txt_path.stem}{ext}"
                if possible_path.exists():
                    img_path = possible_path
                    break

            if not img_path:
                continue  # Skip if there's no matching image

            try:
                # Open image to get real pixel width and height
                with Image.open(img_path) as img:
                    img_w, img_h = img.size

                with open(txt_path, "r") as f:
                    lines = f.readlines()

                new_lines = []
                modified = False

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    class_id = parts[0]
                    coords = [float(x) for x in parts[1:5]]

                    # If any value > 1.0, it's unnormalized absolute pixels
                    if any(c > 1.0 for c in coords):
                        modified = True

                        xmin, ymin, xmax, ymax = coords

                        box_w = xmax - xmin
                        box_h = ymax - ymin
                        cx = xmin + (box_w / 2.0)
                        cy = ymin + (box_h / 2.0)

                        norm_cx = cx / img_w
                        norm_cy = cy / img_h
                        norm_w  = box_w / img_w
                        norm_h  = box_h / img_h

                        # Clip to [0, 1]
                        norm_cx = max(0.0, min(1.0, norm_cx))
                        norm_cy = max(0.0, min(1.0, norm_cy))
                        norm_w  = max(0.0, min(1.0, norm_w))
                        norm_h  = max(0.0, min(1.0, norm_h))

                        new_lines.append(f"{class_id} {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                    else:
                        new_lines.append(line)

                if modified:
                    with open(txt_path, "w") as f:
                        f.writelines(new_lines)
                    fixed_count += 1

            except Exception as e:
                print(f"   [ERROR] Failed to normalize {txt_path.name}: {e}")

        print(f"   └── Successfully normalized {fixed_count} labels inside '{split}'.")

    print("\n✅ Matrix alignment complete! You can now run train.py.")


if __name__ == "__main__":
    normalize_all_labels()
