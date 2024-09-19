import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
train_results = model.train(
    data="./hotdog.yaml", 
    epochs=10, 
    batch=-1, 
    imgsz=640, 
    name="hotdog_model", 
    project="hotdog_detection"
)

path = model.export(format="onnx")