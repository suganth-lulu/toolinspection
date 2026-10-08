"""Train a YOLOv8 defect detector. Usage: python backend/train.py path/to/data.yaml"""
import sys
from ultralytics import YOLO
data = sys.argv[1] if len(sys.argv) > 1 else "data.yaml"
m = YOLO("yolov8n.pt")
m.train(data=data, epochs=50, imgsz=640, batch=16, project="runs", name="forgeguard")
v = m.val()
print(f"mAP50: {v.box.map50:.3f}  mAP50-95: {v.box.map:.3f}  precision: {v.box.mp:.3f}  recall: {v.box.mr:.3f}")
print("Copy runs/forgeguard/weights/best.pt to models/best.pt")
