from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.api import app  # noqa: E402

SAMPLE_IMAGE = ROOT / "data" / "raw" / "images" / "test" / "000010.jpg"


def main() -> None:
    with TestClient(app) as client:
        health_response = client.get("/health")
        print("health:", health_response.status_code, health_response.json())

        with SAMPLE_IMAGE.open("rb") as image_file:
            response = client.post(
                "/predict",
                params={"confidence": 0.25, "image_size": 416},
                files={"file": ("000010.jpg", image_file, "image/jpeg")},
            )

        print("predict:", response.status_code)
        payload = response.json()
        print("filename:", payload["filename"])
        print("image_size:", payload["image_width"], payload["image_height"])
        print("detections:", len(payload["detections"]))
        for detection in payload["detections"][:5]:
            print(detection)


if __name__ == "__main__":
    main()
