from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "data" / "data.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_dataset_config() -> dict:
    with DATA_YAML.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def count_images(path: Path) -> int:
    return sum(1 for item in path.rglob("*") if item.suffix.lower() in IMAGE_EXTENSIONS)


def count_labels(path: Path) -> int:
    return sum(1 for item in path.rglob("*.txt"))


def label_stats(labels_dir: Path) -> Counter:
    classes: Counter[int] = Counter()
    for label_file in labels_dir.rglob("*.txt"):
        for line in label_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            class_id = int(line.split()[0])
            classes[class_id] += 1
    return classes


def paired_files(images_dir: Path, labels_dir: Path) -> tuple[int, int]:
    image_stems = {
        item.relative_to(images_dir).with_suffix("")
        for item in images_dir.rglob("*")
        if item.suffix.lower() in IMAGE_EXTENSIONS
    }
    label_stems = {
        item.relative_to(labels_dir).with_suffix("")
        for item in labels_dir.rglob("*.txt")
    }
    return len(image_stems - label_stems), len(label_stems - image_stems)


def main() -> None:
    config = load_dataset_config()
    dataset_root = ROOT / config["path"]
    names = config["names"]

    print(f"Project root: {ROOT}")
    print(f"Dataset root: {dataset_root}")
    print(f"Classes: {names}")
    print()

    for split in ("train", "val", "test"):
        images_dir = dataset_root / config[split]
        labels_dir = dataset_root / "labels" / split

        images_count = count_images(images_dir) if images_dir.exists() else 0
        labels_count = count_labels(labels_dir) if labels_dir.exists() else 0

        print(f"[{split}]")
        print(f"  images path: {images_dir}")
        print(f"  labels path: {labels_dir}")
        print(f"  images: {images_count}")
        print(f"  labels: {labels_count}")

        if not labels_dir.exists():
            print("  status: missing labels directory")
            print()
            continue

        missing_labels, extra_labels = paired_files(images_dir, labels_dir)
        print(f"  images without labels: {missing_labels}")
        print(f"  labels without images: {extra_labels}")
        print(f"  class instances: {dict(label_stats(labels_dir))}")
        print()


if __name__ == "__main__":
    main()
