from ultralytics import YOLO
import yaml

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

# Load the model
model = YOLO(model_choice)

# Load the dataset config
with open('bdd100k.yaml', 'r') as f:
    dataset_config = yaml.safe_load(f)

print("="*60)
print("MODEL CLASSES")
print("="*60)
print(f"Number of classes: {model.nc}")
print(f"\nClass names:")
for idx, name in model.names.items():
    print(f"  {idx}: {name}")

print("\n" + "="*60)
print("DATASET CLASSES (BDD100K)")
print("="*60)
print(f"Number of classes: {dataset_config['nc']}")
print(f"\nClass names:")
for idx, name in dataset_config['names'].items():
    print(f"  {idx}: {name}")

print("\n" + "="*60)
print("COMPARISON")
print("="*60)

# Check if model classes match dataset classes
model_classes = set(model.names.values())
dataset_classes = set(dataset_config['names'].values())

missing_in_model = dataset_classes - model_classes
extra_in_model = model_classes - dataset_classes

if model.nc != dataset_config['nc']:
    print(f"⚠️  Class count mismatch: Model has {model.nc}, Dataset has {dataset_config['nc']}")
else:
    print(f"✓ Class count matches: {model.nc}")

if missing_in_model:
    print(f"\n⚠️  Classes in dataset but NOT in model:")
    for cls in sorted(missing_in_model):
        print(f"    - {cls}")

if extra_in_model:
    print(f"\n⚠️  Classes in model but NOT in dataset:")
    for cls in sorted(extra_in_model):
        print(f"    - {cls}")

if not missing_in_model and not extra_in_model and model.nc == dataset_config['nc']:
    print("\n✓ All classes match perfectly!")

print("="*60)
