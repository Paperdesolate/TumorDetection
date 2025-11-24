# Tumor Cell Detection – Optimal Model

## Overview
This repository contains our tumor cell detection algorithm, which leverages object detection techniques to automatically extract and recognize tumor cells from imaging data. The approach enables high-throughput, label-free analysis and supports downstream profiling and sorting within the extensional-flow cytometry platform.

## Key Features
- **Object Detection Backbone**: Lightweight detection framework tailored for biomedical imaging.
- **Cell Extraction & Recognition**: Automatic localization and identification of tumor cells with high precision.
- **Self-localization Filters**: Enhance spatial accuracy by dynamically detecting regions of interest.
- **Integration**: Compatible with real-time cytometry workflows for profiling and sorting.

Contents include:
- `best.pt` – the optimal YOLO PyTorch model weights  
- `config.yaml` – training configuration  
- `results.csv` – evaluation metrics summary  

## Usage
Example of running detection with PyTorch YOLO:
```python
from ultralytics import YOLO

# Load the trained model
model = YOLO("run/detect/nbam5/best.pt")

# Run detection on a sample image
results = model("sample_image.png")

# Print detection results
results.show()

