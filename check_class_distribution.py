import os
from pathlib import Path
from collections import defaultdict

# BDD100K classes
BDD_CLASSES = [
    'pedestrian',   # 0
    'rider',        # 1
    'car',          # 2
    'truck',        # 3
    'bus',          # 4
    'train',        # 5
    'motorcycle',   # 6
    'bicycle',      # 7
    'traffic light',# 8
    'traffic sign'  # 9
]

# Count instances per class
class_counts = defaultdict(int)
files_with_class = defaultdict(int)

labels_dir = '/data/datasets/bdd100k-yolopx/labels/val'
label_files = list(Path(labels_dir).glob('*.txt'))

print(f"Analyzing {len(label_files)} label files...\n")

for label_file in label_files:
    with open(label_file, 'r') as f:
        lines = f.readlines()

    classes_in_file = set()
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            class_id = int(parts[0])
            class_counts[class_id] += 1
            classes_in_file.add(class_id)

    for class_id in classes_in_file:
        files_with_class[class_id] += 1

# Print results
print("="*70)
print("CLASS DISTRIBUTION IN VALIDATION SET")
print("="*70)
print(f"{'Class ID':<10} {'Class Name':<18} {'Instances':<12} {'Images':<10}")
print("-"*70)

total_instances = 0
for class_id in range(len(BDD_CLASSES)):
    count = class_counts[class_id]
    files = files_with_class[class_id]
    total_instances += count
    status = "✓" if count > 0 else "✗"
    print(f"{status} {class_id:<8} {BDD_CLASSES[class_id]:<18} {count:<12} {files:<10}")

print("-"*70)
print(f"{'TOTAL':<28} {total_instances:<12}")
print("="*70)

# Check for missing classes
missing_classes = [BDD_CLASSES[i] for i in range(len(BDD_CLASSES)) if class_counts[i] == 0]
if missing_classes:
    print(f"\n⚠️  Classes with NO instances in validation set:")
    for cls in missing_classes:
        print(f"    - {cls}")
else:
    print(f"\n✓ All classes have instances in the validation set!")
