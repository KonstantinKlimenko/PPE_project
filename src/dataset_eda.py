from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "data" / "data.yaml"
REPORTS_DIR = ROOT / "reports"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_config() -> dict:
    with DATA_YAML.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def read_yolo_label(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, x_center, y_center, width, height = line.split()
        rows.append((int(class_id), float(x_center), float(y_center), float(width), float(height)))
    return rows


def collect_annotation_stats(config: dict) -> pd.DataFrame:
    dataset_root = ROOT / config["path"]
    records = []

    for split in ("train", "val", "test"):
        labels_dir = dataset_root / "labels" / split
        for label_path in labels_dir.glob("*.txt"):
            for class_id, x_center, y_center, width, height in read_yolo_label(label_path):
                records.append(
                    {
                        "split": split,
                        "image_id": label_path.stem,
                        "class_id": class_id,
                        "class_name": config["names"][class_id],
                        "x_center": x_center,
                        "y_center": y_center,
                        "width": width,
                        "height": height,
                        "box_area": width * height,
                    }
                )

    return pd.DataFrame(records)


def save_class_distribution(df: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)

    summary = (
        df.groupby(["split", "class_name"])
        .size()
        .reset_index(name="objects")
        .sort_values(["split", "objects"], ascending=[True, False])
    )
    summary.to_csv(REPORTS_DIR / "class_distribution.csv", index=False)

    pivot = summary.pivot(index="class_name", columns="split", values="objects").fillna(0)
    pivot.plot(kind="bar", figsize=(9, 5))
    plt.title("Class distribution by split")
    plt.xlabel("Class")
    plt.ylabel("Objects")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "class_distribution.png", dpi=150)
    plt.close()


def yolo_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    box_width = width * image_width
    box_height = height * image_height
    x1 = int((x_center * image_width) - box_width / 2)
    y1 = int((y_center * image_height) - box_height / 2)
    x2 = int((x_center * image_width) + box_width / 2)
    y2 = int((y_center * image_height) + box_height / 2)
    return x1, y1, x2, y2


def draw_sample_images(config: dict, split: str = "train", samples: int = 6) -> None:
    dataset_root = ROOT / config["path"]
    images_dir = dataset_root / config[split]
    labels_dir = dataset_root / "labels" / split
    output_dir = REPORTS_DIR / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [
        path for path in images_dir.glob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    selected_paths = random.sample(image_paths, k=min(samples, len(image_paths)))

    colors = {
        0: "green",
        1: "orange",
        2: "red",
        3: "royalblue",
    }

    for image_path in selected_paths:
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        image_width, image_height = image.size
        label_path = labels_dir / f"{image_path.stem}.txt"

        for class_id, x_center, y_center, width, height in read_yolo_label(label_path):
            x1, y1, x2, y2 = yolo_to_xyxy(
                x_center, y_center, width, height, image_width, image_height
            )
            color = colors.get(class_id, (255, 255, 255))
            label = config["names"][class_id]
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            draw.text((x1, max(y1 - 14, 0)), label, fill=color, font=font)

        image.save(output_dir / image_path.name)


def main() -> None:
    config = load_config()
    df = collect_annotation_stats(config)

    print("Total objects:", len(df))
    print("Objects by class:", dict(Counter(df["class_name"])))
    print("Objects by split:")
    print(df.groupby("split").size())

    save_class_distribution(df)
    draw_sample_images(config)

    print(f"Saved reports to: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
