import os
import xml.etree.ElementTree as ET
from pathlib import Path

# ─────────────────────────────── CONFIG ──────────────────────────────────────
XML_DIR = Path("annotations/")
TXT_OUT_DIR = Path("labels") # We will output clean text labels here
TXT_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Update this map to match your exact data.yaml indices!
CLASS_MAPPING = {
    "fox" : 0,
    "leopard" : 1,
    "sheep": 2
}

def convert_voc_to_yolo():
    print("============================================================")
    print("  Executing Pascal VOC (XML) to YOLO (TXT) Converter")
    print("============================================================\n")

    converted = 0

    for xml_file in XML_DIR.glob("*.xml"):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Extract actual image dimensions from XML tags
            size = root.find("size")
            img_w = float(size.find("width").text)
            img_h = float(size.find("height").text)

            yolo_lines = []

            for obj in root.findall("object"):
                cls_name = obj.find("name").text.lower().strip()
                if cls_name not in CLASS_MAPPING:
                    continue # Skip classes you don't care about

                class_id = CLASS_MAPPING[cls_name]
                bndbox = obj.find("bndbox")

                # Extract absolute bounding box coordinates
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)

                # Math conversion to center points, widths, and heights
                box_w = xmax - xmin
                box_h = ymax - ymin
                cx = xmin + (box_w / 2.0)
                cy = ymin + (box_h / 2.0)

                # Normalize values by full canvas boundaries
                norm_cx = cx / img_w
                norm_cy = cy / img_h
                norm_w  = box_w / img_w
                norm_h  = box_h / img_h

                yolo_lines.append(f"{class_id} {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}\n")

            # Save the text file with the identical name stem
            txt_path = TXT_OUT_DIR / f"{xml_file.stem}.txt"
            with open(txt_path, "w") as f:
                f.writelines(yolo_lines)
            converted += 1

        except Exception as e:
            print(f"   [ERROR] Failed processing {xml_file.name}: {e}")

    print(f"\n✅ Successfully converted {converted} XML files into YOLO tracking files.")

if __name__ == "__main__":
    convert_voc_to_yolo()