from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="/content/dataset/data.yaml", 
    epochs=30,  
    imgsz=640, 
    batch=8,  
    device="cuda",  
    save_period=5
)
from google.colab import drive
drive.mount('/content/drive')
!unzip /content/drive/MyDrive/Fruit_Ripening_Process.v2i.yolov8-obb.zip -d /content/dataset
!pip install ultralytics
!pip install roboflow
!pip install supervision
from roboflow import Roboflow
import json

rf = Roboflow(api_key="PmxZX6G7dTur7WCKDRa4")
project = rf.workspace().project("fruit-ripening-process")
model = project.version(2).model

image_path = "/content/dataset/test/images/musa-acuminata-unripe-538c346f-2653-11ec-93c0-d8c4975e38aa---Copy_jpg.rf.c53c6e678225bd031fb13e0a125c83c6.jpg"
result = model.predict(image_path, confidence=40, overlap=30).json()

bad_labels = {"rotten", "overripe"}
is_bad = False

for prediction in result["predictions"]:
    fruit_label = prediction["class"]
    print(fruit_label)
    if fruit_label in bad_labels:
        is_bad = True
        break

if is_bad:
    print("Bad")
else:
    print("Good")
