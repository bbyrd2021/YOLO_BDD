from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import numpy as np

choice_made = False

while not choice_made:
    print("**Choose model to evaluate:**")
    print("     1. yolov10-bdd-finetune.pt")
    print("     2. yolov10-bdd-vanilla.pt")

    choice = input("")

    if choice == '1':
        model_choice = './model/yolov10-bdd-finetune.pt'
        print("you choose yolov10-bdd-finetune.pt")
        choice_made = True
    elif choice == '2':
        model_choice = './model/yolov10-bdd-vanilla.pt'
        print("you choose yolov10-bdd-vanilla.pt")
        choice_made = True
    else:
        print("please try again")

    

# Load model
model = YOLO(model_choice)

# Run validation on BDD100K validation set
results = model.val(
    data='bdd100k.yaml',  # dataset configuration
    imgsz=640,            # image size
    batch=16,             # batch size
    conf=0.001,           # confidence threshold
    iou=0.6,              # NMS IoU threshold
    device=0,             # GPU device (0 for first GPU, 'cpu' for CPU)
    save_json=True,       # save results to JSON
    save_hybrid=False,    # save hybrid labels
    plots=True,           # save plots
)

# Calculate raw average IoU per class and overall
print("\nCalculating raw average IoU values...")

def box_iou(box1, box2):
    """Calculate IoU between two boxes in xyxy format"""
    # Get intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate areas
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

# Load validation dataset to get image paths and labels
from pathlib import Path
import yaml

# Load the YAML to get the base path
with open('bdd100k.yaml', 'r') as f:
    yaml_data = yaml.safe_load(f)

base_path = Path(yaml_data['path'])
val_relative = yaml_data['val']

# Construct full paths
val_img_dir = base_path / val_relative
label_dir = base_path / 'labels' / 'val'

# Debug: print paths
print(f"Base path: {base_path}")
print(f"Val images dir: {val_img_dir}")
print(f"Label dir: {label_dir}")

# Get list of images
image_files = sorted(list(val_img_dir.glob('*.jpg')) + list(val_img_dir.glob('*.png')))

print(f"Found {len(image_files)} validation images")

# Initialize per-class IoU storage
class_ious = {i: [] for i in range(len(model.names))}
all_ious = []

# Process a subset of validation images (for efficiency)
num_to_process = min(len(image_files), 1000)
print(f"Processing {num_to_process} images for raw IoU calculation...")

for img_idx, img_path in enumerate(image_files[:num_to_process]):
    if img_idx % 100 == 0:
        print(f"  Processing image {img_idx}...")

    # Get corresponding label file
    # Handle both direct paths and glob results
    if isinstance(img_path, str):
        img_path = Path(img_path)

    # Determine label path based on image path structure
    if 'images' in img_path.parts:
        # Standard YOLO structure: replace 'images' with 'labels'
        label_parts = list(img_path.parts)
        for i, part in enumerate(label_parts):
            if part == 'images':
                label_parts[i] = 'labels'
                break
        label_path = Path(*label_parts).with_suffix('.txt')
    else:
        # Fallback to label_dir
        label_path = label_dir / img_path.with_suffix('.txt').name

    if not label_path.exists():
        if img_idx == 0:
            print(f"  Warning: Label file not found: {label_path}")
        continue

    # Read ground truth boxes
    gt_boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                # Convert from YOLO format (x_center, y_center, w, h) to (x1, y1, x2, y2)
                x_center, y_center, w, h = map(float, parts[1:5])
                x1 = x_center - w/2
                y1 = y_center - h/2
                x2 = x_center + w/2
                y2 = y_center + h/2
                gt_boxes.append((cls, [x1, y1, x2, y2]))

    if not gt_boxes:
        continue

    # Run prediction
    pred_results = model.predict(str(img_path), conf=0.001, iou=0.6, verbose=False)

    if len(pred_results) == 0 or pred_results[0].boxes is None:
        continue

    pred = pred_results[0]

    # Get prediction boxes (normalized)
    if len(pred.boxes) == 0:
        continue

    pred_boxes = pred.boxes.xyxyn.cpu().numpy()  # normalized xyxy
    pred_classes = pred.boxes.cls.cpu().numpy().astype(int)
    pred_confs = pred.boxes.conf.cpu().numpy()

    # Debug output for first image
    if img_idx == 0:
        print(f"  First image: {img_path.name}")
        print(f"    GT boxes: {len(gt_boxes)}, Predictions: {len(pred_boxes)}")
        if len(gt_boxes) > 0:
            print(f"    Sample GT box: class={gt_boxes[0][0]}, box={gt_boxes[0][1]}")
        if len(pred_boxes) > 0:
            print(f"    Sample pred box: class={pred_classes[0]}, box={pred_boxes[0]}")

    # Match predictions to ground truth and calculate IoU
    matched_gt = set()

    for pred_idx, (pred_box, pred_cls, pred_conf) in enumerate(zip(pred_boxes, pred_classes, pred_confs)):
        best_iou = 0
        best_gt_idx = -1

        # Find best matching ground truth box of same class
        for gt_idx, (gt_cls, gt_box) in enumerate(gt_boxes):
            if gt_cls == pred_cls and gt_idx not in matched_gt:
                iou = box_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

        # If we found a match with IoU > 0.5, record it
        if best_iou >= 0.5 and best_gt_idx != -1:
            matched_gt.add(best_gt_idx)
            class_ious[pred_cls].append(best_iou)
            all_ious.append(best_iou)

    # Debug output for first image
    if img_idx == 0 and len(all_ious) > 0:
        print(f"    Matched detections in first image: {len(all_ious)}, avg IoU: {np.mean(all_ious):.4f}")

# Calculate average IoU per class
avg_class_iou = {}
for cls_id, ious in class_ious.items():
    if ious:
        avg_class_iou[cls_id] = np.mean(ious)
    else:
        avg_class_iou[cls_id] = 0.0

# Calculate overall average IoU
overall_avg_iou = np.mean(all_ious) if all_ious else 0.0

print(f"Raw average IoU calculation complete. Processed {len(all_ious)} matched detections.")

# Print metrics
print("\n" + "="*60)
print("VALIDATION RESULTS - OVERALL METRICS")
print("="*60)
print(f"mAP50 (Accuracy@0.5 IoU):    {results.results_dict['metrics/mAP50(B)']:.4f}")
print(f"mAP50-95:                    {results.results_dict['metrics/mAP50-95(B)']:.4f}")
print(f"Precision:                   {results.results_dict['metrics/precision(B)']:.4f}")
print(f"Recall:                      {results.results_dict['metrics/recall(B)']:.4f}")

# Calculate F1 score
precision = results.results_dict['metrics/precision(B)']
recall = results.results_dict['metrics/recall(B)']
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
print(f"F1 Score:                    {f1:.4f}")
print(f"Raw Average IoU (overall):   {overall_avg_iou:.4f}")
print("="*60)

# Export per-class metrics to CSV
print("\nExporting per-class metrics to CSV...")

# Get per-class metrics
class_names = list(model.names.values())
per_class_metrics = []

# Extract metrics for each class from the results
# results.box contains DetMetrics with arrays for per-class values
for idx, class_name in enumerate(class_names):
    precision = float(results.box.p[idx])
    recall = float(results.box.r[idx])
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    ap50 = float(results.box.ap50[idx])  # This is accuracy at 50% IoU threshold
    ap = float(results.box.ap[idx])
    raw_iou = avg_class_iou.get(idx, 0.0)

    metrics = {
        'Class': class_name,
        'Class_ID': idx,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'AP@0.5 (Acc@50% IoU)': ap50,
        'AP@0.5:0.95': ap,
        'Raw Avg IoU': raw_iou,
    }
    per_class_metrics.append(metrics)

# Add overall metrics as the last row
overall_precision = float(results.results_dict['metrics/precision(B)'])
overall_recall = float(results.results_dict['metrics/recall(B)'])
overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

overall_metrics = {
    'Class': 'all',
    'Class_ID': -1,
    'Precision': overall_precision,
    'Recall': overall_recall,
    'F1': overall_f1,
    'AP@0.5 (Acc@50% IoU)': float(results.results_dict['metrics/mAP50(B)']),
    'AP@0.5:0.95': float(results.results_dict['metrics/mAP50-95(B)']),
    'Raw Avg IoU': overall_avg_iou,
}
per_class_metrics.append(overall_metrics)

# Create DataFrame and save to CSV
df = pd.DataFrame(per_class_metrics)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
csv_filename = f'bdd100k_validation_metrics_{timestamp}.csv'
df.to_csv(csv_filename, index=False)

print(f"Per-class metrics saved to: {csv_filename}")
print(f"\nCSV Preview:")
print(df.to_string(index=False))

# Print detailed per-class breakdown for debugging
print("\n" + "="*60)
print("PER-CLASS BREAKDOWN (for debugging)")
print("="*60)
print(f"{'Class':<20} {'Prec':<8} {'Recall':<8} {'F1':<8} {'AP@0.5':<8} {'Avg IoU':<8}")
print("-"*70)
for idx, class_name in enumerate(class_names):
    p = results.box.p[idx]
    r = results.box.r[idx]
    f1_val = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
    ap50_val = results.box.ap50[idx]
    raw_iou = avg_class_iou.get(idx, 0.0)
    print(f"{class_name:<20} {p:<8.4f} {r:<8.4f} {f1_val:<8.4f} {ap50_val:<8.4f} {raw_iou:<8.4f}")
print("="*70)

print("\n" + "="*60)
print("NOTES FOR PAPER COMPARISON:")
print("="*60)
print("- mAP50 (AP@0.5) is the standard accuracy metric at 50% IoU threshold")
print("- This metric counts predictions with IoU >= 0.5 as correct")
print("- mAP50-95 averages performance across IoU thresholds 0.5 to 0.95")
print("- F1 Score balances precision and recall: F1 = 2*(P*R)/(P+R)")
print("- Raw Avg IoU is the mean IoU value of all matched detections (IoU >= 0.5)")
print("  * Shows average overlap quality for correct predictions")
print("  * Higher values indicate tighter bounding box localization")
print("- Compare your mAP50 and mAP50-95 values to published paper results")
print("="*60)

# Try to access confusion matrix for additional IoU insights
print("\n" + "="*60)
print("ADDITIONAL METRICS (if available)")
print("="*60)
try:
    if hasattr(results, 'confusion_matrix') and results.confusion_matrix is not None:
        print("Confusion matrix data available")
        # You can add more detailed confusion matrix analysis here if needed
    if hasattr(results, 'speed'):
        print(f"Inference speed: {results.speed}")
    if hasattr(results.box, 'all_ap'):
        print(f"All AP values available for detailed analysis")
except Exception as e:
    print(f"Note: Some extended metrics not available: {e}")
print("="*60)

