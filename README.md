
___

# L2CS-Net

The official PyTorch implementation of L2CS-Net for gaze estimation and tracking.

## Installation

Install package with the following:

```
git clone https://github.com/acse-hz923/L2CS-Net.git
```
```
pip install git+https://github.com/edavalosanaya/L2CS-Net.git@main
```

## Usage

Detect face and predict gaze from webcam

```python
from l2cs import Pipeline, render
import cv2

gaze_pipeline = Pipeline(
    weights=CWD / 'models' / 'L2CSNet_gaze360.pkl',
    arch='ResNet50',
    device=torch.device('cpu') # or 'gpu'
)
 
cap = cv2.VideoCapture(cam)
_, frame = cap.read()    

# Process frame and visualize
results = gaze_pipeline.step(frame)
frame = render(frame, results)
```

## Demo
* Download the pre-trained models from [here](https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd?usp=sharing) and Store it to *models/*.
*  Run:
```
 python demo.py \
 --snapshot models/L2CSNet_gaze360.pkl \
 --gpu 0 \
 --cam 0 \
```
This means the demo will run using *L2CSNet_gaze360.pkl* pretrained model

