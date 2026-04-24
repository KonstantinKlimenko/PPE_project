from __future__ import annotations

from contextlib import asynccontextmanager
import io
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "Ultralytics"))

from ultralytics import YOLO  # noqa: E402


DEFAULT_MODEL_PATH = ROOT / "models" / "ppe_yolo11n_baseline.pt"
MODEL_PATH = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    box: list[float]


class PredictionResponse(BaseModel):
    filename: str
    image_width: int
    image_height: int
    detections: list[Detection]


model: YOLO | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model weights not found: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    try:
        yield
    finally:
        model = None


app = FastAPI(
    title="Construction PPE Detection API",
    description="YOLO-based API for detecting helmets, safety vests, and heads.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_path": str(MODEL_PATH)}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    confidence: float = 0.25,
    image_size: int = 416,
) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file")

    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    result = model.predict(image, imgsz=image_size, conf=confidence, verbose=False)[0]

    detections = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        detections.append(
            Detection(
                class_id=class_id,
                class_name=result.names[class_id],
                confidence=round(float(box.conf[0]), 4),
                box=[round(value, 2) for value in box.xyxy[0].tolist()],
            )
        )

    return PredictionResponse(
        filename=file.filename or "uploaded_image",
        image_width=image.width,
        image_height=image.height,
        detections=detections,
    )
