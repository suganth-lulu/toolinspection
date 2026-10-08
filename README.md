# ForgeGuard AI — real YOLOv8 defect inspection

## 1. Install
    python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
    pip install -r requirements.txt

## 2. Train a model (use Google Colab GPU if your laptop is slow)
1. Download a defect dataset in YOLO format (e.g. NEU-DET steel surface defects) and make `data.yaml`:

       path: /full/path/to/dataset
       train: images/train
       val: images/val
       names: {0: crack, 1: inclusion, 2: patches, 3: pitted_surface, 4: rolled-in_scale, 5: scratches}
2. `python backend/train.py data.yaml`  (prints mAP50, precision, recall — quote these to judges)
3. Copy `runs/forgeguard/weights/best.pt` to `models/best.pt`

## 3. Run
    uvicorn backend.main:app --port 8000
Open http://localhost:8000 (camera works on localhost). The header shows "YOLOV8 REAL MODEL" when the model is loaded.
If `models/best.pt` is missing, the site falls back to the rule-based demo analysis and says so.

## Honest limits
- Detects only the defect types it was trained on; show products that match your dataset.
- Sensors and demo history in the UI are simulated. Risk score is a prioritisation score, not a validated metric.
- Not included yet: anomaly heatmaps (PatchCore), sensor API/correlation, PDF report.
