from ultralytics import YOLO
import cv2
import os

model = YOLO("best_yolov8s.pt")
# result = model.predict(["carnum.jpeg","carnum2.jpeg","carnum3.png","13012.png"], device="mps", save=True, conf=0.4, iou=0.5)
# result = model.predict(["carnum.jpeg","carnum2.jpeg","carnum3.png","13012.png"], device="mps", save=True, conf=0.4, iou=0.6)
result = model.predict(["carnum.jpeg","carnum2.jpeg","carnum3.png","13012.png"], device="mps", save=True, conf=0.4, iou=0.7)