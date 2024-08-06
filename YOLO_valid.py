from ultralytics import YOLO
import cv2
import os

model = YOLO("best_YOLOv10m.pt")

validation_result = model.val(data="datasets/data.yaml", imgsz=640, batch=16, conf=0.35, iou=0.6, device="mps")