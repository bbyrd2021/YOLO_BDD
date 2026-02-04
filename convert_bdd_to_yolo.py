import json
import os
from pathlib import Path
from PIL import Image

# BDD100K detection classes (10 classes)
BDD_CLASSES = [
    'pedestrian',
    'rider',
    'car',
    'truck',
    'bus',
    'train',
    'motorcycle',
    'bicycle',
    'traffic light',
    'traffic sign'
]

# Mapping from BDD100K JSON category names to standard class names
CATEGORY_MAPPING = {
    'person': 'pedestrian',
    'rider': 'rider',
    'car': 'car',
    'truck': 'truck',
    'bus': 'bus',
    'train': 'train',
    'motor': 'motorcycle',
    'bike': 'bicycle',
    'traffic light': 'traffic light',
    'traffic sign': 'traffic sign'
}

def convert_box_to_yolo(box2d, img_width, img_height):
    """Convert BDD100K box2d to YOLO format (normalized)"""
    x1, y1, x2, y2 = box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2']

    # Calculate center, width, height
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1

    # Normalize to [0, 1]
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height

    return x_center_norm, y_center_norm, width_norm, height_norm

def convert_json_to_yolo(json_path, output_dir, img_dir):
    """Convert a single BDD100K JSON file to YOLO format"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Get image name
    img_name = data['name']
    img_path = os.path.join(img_dir, f"{img_name}.jpg")

    # Get image dimensions
    if not os.path.exists(img_path):
        print(f"Warning: Image not found: {img_path}")
        return

    with Image.open(img_path) as img:
        img_width, img_height = img.size

    # Process annotations
    yolo_annotations = []
    for frame in data['frames']:
        for obj in frame['objects']:
            category = obj.get('category')

            # Skip if no box2d (poly2d objects)
            if 'box2d' not in obj:
                continue

            # Map BDD100K category to standard class name
            if category not in CATEGORY_MAPPING:
                continue  # Skip non-detection categories (lane/area)

            mapped_category = CATEGORY_MAPPING[category]
            class_id = BDD_CLASSES.index(mapped_category)
            box2d = obj['box2d']

            # Convert to YOLO format
            x_center, y_center, width, height = convert_box_to_yolo(
                box2d, img_width, img_height
            )

            yolo_annotations.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

    # Write to txt file
    output_path = os.path.join(output_dir, f"{img_name}.txt")
    with open(output_path, 'w') as f:
        f.write('\n'.join(yolo_annotations))

    return len(yolo_annotations)

def main():
    # Paths
    json_dir = '/data/datasets/bdd100k-yolopx/det_annotations/val'
    img_dir = '/data/datasets/bdd100k-yolopx/images/val'
    output_dir = '/data/datasets/bdd100k-yolopx/labels/val'

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Convert all JSON files
    json_files = list(Path(json_dir).glob('*.json'))
    total_annotations = 0

    print(f"Converting {len(json_files)} JSON files...")
    for i, json_path in enumerate(json_files):
        num_annotations = convert_json_to_yolo(json_path, output_dir, img_dir)
        if num_annotations is not None:
            total_annotations += num_annotations

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(json_files)} files...")

    print(f"\nConversion complete!")
    print(f"Total files: {len(json_files)}")
    print(f"Total annotations: {total_annotations}")
    print(f"Output directory: {output_dir}")

if __name__ == '__main__':
    main()
