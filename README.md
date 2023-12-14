# YOLOv8_Car_Number_Plate_Detection

## INTRODUCTION :
This repository provides a comprehensive guide and codebase for training a car number plate detection model using YOLOv8n on a GPU. We will be training a yolov8n model 21,173 images for training, 1019 test images and 2046 validation images for 100 epochs on rtx 3060 gpu(12gb ram) which took me 3.463 hours on RTX 3060 gpu (12gb ram). We will be using one more model which is pretrained YOLOv8n for detecting vehicles and then we will use our custom trained model to detect license plate on those vehicles 
## REQUIREMENTS :
* Requirements :
* Python (3.8 or higher)
* CUDA-enabled GPU
* CUDA version of pytorch install

## Installation :
(For Windows)
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
visit https://pytorch.org/get-started/locally/ for different OS installation
```
pip install ultralytics
```
For dataset use this link https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4
##
Code for Training :
```
from ultralytics import YOLO
# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from YAML
# Train the model
if __name__ == '__main__':      
    results = model.train(data='data.yaml', resume=True, epochs=100) #path for config file should be given

```
## CODE :

After training the model we will perform objection detection on the sample video for only vehicles like car, motorbike, bus, truck. Then we will keep track of those vehicle with car_ID. After that we will input the cropped image of vehicle to the license plate detection model. The util.py script contains ocr code for extracting the text of the license plate and to give us license plate confidence score. The complete results are created and saved in "test.csv" file. After that we will use add_missing_data script to modify the text.csv file to fill in the missing "frame_nmr" coloumn where no license plate was detected.




