from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize YOLO training results.")
    parser.add_argument(
        "--run",
        type=Path,
        default=ROOT / "runs" / "detect" / "ppe_yolo11n_pretrained_5ep_frac10",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = args.run / "results.csv"

    df = pd.read_csv(results_path)
    df.columns = [column.strip() for column in df.columns]
    best_row = df.loc[df["metrics/mAP50(B)"].idxmax()]

    print(f"Run: {args.run}")
    print(f"Best epoch by mAP50: {int(best_row['epoch'])}")
    print(f"Precision: {best_row['metrics/precision(B)']:.3f}")
    print(f"Recall: {best_row['metrics/recall(B)']:.3f}")
    print(f"mAP50: {best_row['metrics/mAP50(B)']:.3f}")
    print(f"mAP50-95: {best_row['metrics/mAP50-95(B)']:.3f}")


if __name__ == "__main__":
    main()
