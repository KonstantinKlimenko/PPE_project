from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "Ultralytics"))

from ultralytics import YOLO  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO inference on an image.")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "runs" / "detect" / "ppe_yolo_sanity_scratch" / "weights" / "best.pt",
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "predict")
    parser.add_argument("--name", type=str, default="sample_prediction")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO(args.model)
    results = model.predict(
        source=str(args.source),
        imgsz=args.imgsz,
        conf=args.conf,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        save=True,
    )

    for result in results:
        print(f"Image: {result.path}")
        print(f"Detections: {len(result.boxes)}")
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])
            xyxy = [round(value, 2) for value in box.xyxy[0].tolist()]
            print(f"  {class_name}: conf={confidence:.3f}, box={xyxy}")


if __name__ == "__main__":
    main()
