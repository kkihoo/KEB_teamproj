from ultralytics import YOLO
import cv2
import os

model = YOLO('runs/detect/train/weights/last.pt')

# train_result = model.train(data="datasets/data.yaml", epochs = 50, imgsz=640, lr0=0.00025, optimizer='AdamW', save=True, save_period=5, patience=10)
train_result = model.train(resume=True, device="mps")