from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "Ultralytics"))

from ultralytics import YOLO  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO model for PPE detection.")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "data.yaml")
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=0.05)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "detect")
    parser.add_argument("--name", type=str, default="ppe_yolo_sanity")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        fraction=args.fraction,
        amp=not args.no_amp,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
