# YOLO BDD100K Object Detection

YOLOv10 object detection model trained and evaluated on the BDD100K dataset for autonomous driving scenarios.

## Overview

This project implements object detection for autonomous driving using YOLOv10 on the BDD100K dataset. The BDD100K dataset contains diverse driving scenes with 10 object classes including vehicles, pedestrians, traffic signs, and traffic lights.

## Features

- Data conversion from BDD100K format to YOLO format
- Model evaluation with comprehensive metrics
- Support for both vanilla and fine-tuned YOLOv10 models
- Per-class performance analysis
- IoU calculation and validation metrics

## Object Classes

The model detects 10 object classes:
1. Pedestrian
2. Rider
3. Car
4. Truck
5. Bus
6. Train
7. Motorcycle
8. Bicycle
9. Traffic Light
10. Traffic Sign

## Requirements

```bash
pip install ultralytics pillow pandas numpy pyyaml
```

## Dataset Setup

1. Download the BDD100K dataset from [https://www.bdd100k.com/](https://www.bdd100k.com/)
2. Update the `path` in [bdd100k.yaml](bdd100k.yaml) to point to your dataset location
3. Convert annotations to YOLO format:

```bash
python convert_bdd_to_yolo.py
```

## Project Structure

```
YOLO_BDD/
├── bdd100k.yaml              # Dataset configuration
├── convert_bdd_to_yolo.py    # Convert BDD100K to YOLO format
├── check_class_distribution.py # Analyze class distribution
├── check_classes.py          # Verify classes in dataset
├── eval.py                   # Model evaluation script
├── model/                    # Model weights (not tracked in git)
│   ├── yolov10-bdd-vanilla.pt
│   └── yolov10-bdd-finetune.pt
└── runs/                     # Training/validation results
```

## Usage

### Convert Dataset

```bash
python convert_bdd_to_yolo.py
```

Edit the paths in the script to match your dataset location.

### Evaluate Model

```bash
python eval.py
```

The script will prompt you to choose between:
- `yolov10-bdd-finetune.pt` - Fine-tuned model
- `yolov10-bdd-vanilla.pt` - Vanilla model

### Check Class Distribution

```bash
python check_class_distribution.py
```

## Evaluation Metrics

The evaluation script provides:
- **mAP50**: Mean Average Precision at IoU threshold 0.5
- **mAP50-95**: Mean Average Precision averaged over IoU thresholds 0.5 to 0.95
- **Precision**: Ratio of correct predictions to total predictions
- **Recall**: Ratio of correct predictions to total ground truth objects
- **F1 Score**: Harmonic mean of precision and recall
- **Raw Average IoU**: Mean IoU of matched detections
- **Per-class metrics**: Individual performance for each object class

Results are exported to CSV files with timestamps.

## Model Files

Model weights are not included in this repository due to their size. You can:
- Train your own model using YOLOv10
- Download pre-trained weights and place them in the `model/` directory

## Configuration

Edit [bdd100k.yaml](bdd100k.yaml) to configure:
- Dataset paths
- Train/validation splits
- Class names and IDs

## Results

Validation results are saved to:
- CSV files: `bdd100k_validation_metrics_TIMESTAMP.csv`
- Plots: `runs/detect/val*/` directories
- Confusion matrices, precision-recall curves, F1 curves

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics YOLOv10](https://github.com/ultralytics/ultralytics)
- [BDD100K Dataset](https://www.bdd100k.com/)

## Citation

If you use this code in your research, please cite the BDD100K dataset:

```bibtex
@inproceedings{bdd100k,
  title={BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning},
  author={Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, Wenqi and Chen, Yingying and Liu, Fangchen and Madhavan, Vashisht and Darrell, Trevor},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```
