"""ForgeGuard AI backend: real YOLOv8 inference + SQLite log. Serves the frontend too."""
import io, os, sqlite3, time
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("FG_MODEL", ROOT / "models" / "best.pt"))
CONF = float(os.getenv("FG_CONF", "0.35"))   # defect threshold (tune on your validation set)

app = FastAPI(title="ForgeGuard AI")
model = None
try:
    from ultralytics import YOLO
    if MODEL_PATH.exists():
        model = YOLO(str(MODEL_PATH))
except Exception as e:
    print("Model load failed:", e)

db = sqlite3.connect(ROOT / "forgeguard.db", check_same_thread=False)
db.execute("""create table if not exists inspections(id integer primary key, ts text,
  result text, defect text, conf real, severity text, risk int)""")

def severity(area, conf):
    if area > 0.08 or (area > 0.03 and conf > 0.85): return "HIGH"
    return "MEDIUM" if area > 0.02 else "LOW"

@app.get("/health")
def health():
    return {"model_loaded": model is not None, "model": MODEL_PATH.name}

@app.post("/inspect")
async def inspect(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse({"error": "model_not_loaded"}, status_code=503)
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    W, H = img.size
    t = time.time()
    r = model.predict(img, conf=0.05, verbose=False)[0]
    ms = int((time.time() - t) * 1000)
    dets, other = [], 0.0
    for b in r.boxes:
        c = float(b.conf[0])
        if c < CONF:
            other = max(other, c); continue
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        area = (x2 - x1) * (y2 - y1) / (W * H)
        dets.append({"name": r.names[int(b.cls[0])], "conf": round(c * 100, 1),
                     "box": [x1 / W, y1 / H, (x2 - x1) / W, (y2 - y1) / H],
                     "severity": severity(area, c), "area": round(area, 4)})
    dets.sort(key=lambda d: -d["conf"])
    top = dets[0] if dets else None
    # Prioritisation score (not a validated metric): confidence + defect size
    risk = min(99, int(35 + top["conf"] * 0.3 + min(top["area"] * 400, 30))) if top else 10
    db.execute("insert into inspections(ts,result,defect,conf,severity,risk) values(?,?,?,?,?,?)",
               (datetime.now().isoformat(timespec="seconds"), "FAIL" if top else "PASS",
                top["name"] if top else None, top["conf"] if top else None,
                top["severity"] if top else "LOW", risk))
    db.commit()
    return {"detections": dets, "risk": risk, "ms": ms,
            "pass_conf": round((1 - other) * 100, 1)}

@app.get("/history")
def history():
    rows = db.execute("select ts,result,defect,conf,severity,risk from inspections order by id desc limit 50").fetchall()
    return [dict(zip(["ts", "result", "defect", "conf", "severity", "risk"], r)) for r in rows]

app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="web")
