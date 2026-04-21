from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "Ultralytics"))

from ultralytics import YOLO  # noqa: E402


DEFAULT_MODEL_PATH = (
    ROOT
    / "runs"
    / "detect"
    / "ppe_yolo11n_pretrained_5ep_frac10"
    / "weights"
    / "best.pt"
)
MODEL_PATH = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
PREDICTIONS_DIR = ROOT / "reports" / "predictions"
FEEDBACK_PATH = ROOT / "reports" / "feedback.csv"

COLORS = {
    "helmet": "#22c55e",
    "vest": "#f59e0b",
    "head": "#ef4444",
    "person": "#3b82f6",
}


st.set_page_config(
    page_title="Контроль СИЗ",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1180px;
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    .stButton > button {
        width: 100%;
        border-radius: 6px;
        font-weight: 600;
    }
    .metric-row {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin: 10px 0 18px;
    }
    .metric-box {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px 14px;
        background: #ffffff;
    }
    .metric-label {
        font-size: 13px;
        color: #6b7280;
        margin-bottom: 2px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #111827;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def draw_detections(image: Image.Image, detections: list[dict]) -> Image.Image:
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        class_name = detection["class_name"]
        confidence = detection["confidence"]
        color = COLORS.get(class_name, "#ffffff")
        label = f"{class_name} {confidence:.2f}"

        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        label_y = max(y1 - text_height - 8, 0)
        draw.rectangle(
            (x1, label_y, x1 + text_width + 8, label_y + text_height + 6),
            fill=color,
        )
        draw.text((x1 + 4, label_y + 3), label, fill="#ffffff", font=font)

    return annotated


def predict_image(model: YOLO, image: Image.Image, confidence: float, image_size: int) -> list[dict]:
    result = model.predict(image, imgsz=image_size, conf=confidence, verbose=False)[0]

    detections = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = result.names[class_id]
        detections.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(float(box.conf[0]), 4),
                "box": [round(value, 2) for value in box.xyxy[0].tolist()],
            }
        )
    return detections


def detections_to_frame(detections: list[dict]) -> pd.DataFrame:
    if not detections:
        return pd.DataFrame(columns=["class", "confidence", "x1", "y1", "x2", "y2"])

    rows = []
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        rows.append(
            {
                "class": detection["class_name"],
                "confidence": detection["confidence"],
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
    return pd.DataFrame(rows)


def image_to_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def create_prediction_id(image_bytes: bytes, confidence: float, image_size: int) -> str:
    digest_source = image_bytes + f"{confidence:.2f}_{image_size}".encode("utf-8")
    prediction_key = hashlib.sha1(digest_source).hexdigest()
    session_key = f"prediction_id_{prediction_key}"

    if session_key in st.session_state:
        return st.session_state[session_key]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prediction_id = f"{timestamp}_{prediction_key[:10]}"
    st.session_state[session_key] = prediction_id
    return prediction_id


def save_prediction_artifacts(
    prediction_id: str,
    original_image: Image.Image,
    annotated_image: Image.Image,
    detections: list[dict],
    source_filename: str,
    confidence: float,
    image_size: int,
) -> Path:
    output_dir = PREDICTIONS_DIR / prediction_id
    output_dir.mkdir(parents=True, exist_ok=True)

    original_image.save(output_dir / "original.png")
    annotated_image.save(output_dir / "annotated.png")

    metadata = {
        "prediction_id": prediction_id,
        "source_filename": source_filename,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "confidence_threshold": confidence,
        "image_size": image_size,
        "detections": detections,
    }
    (output_dir / "prediction.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_dir


def append_feedback(prediction_id: str, feedback: str, comment: str) -> None:
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = FEEDBACK_PATH.exists()

    with FEEDBACK_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["created_at", "prediction_id", "feedback", "comment"],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "prediction_id": prediction_id,
                "feedback": feedback,
                "comment": comment,
            }
        )


st.title("Контроль средств индивидуальной защиты")

st.sidebar.header("Настройки")
model_path = st.sidebar.text_input("Модель", value=str(MODEL_PATH))
confidence = st.sidebar.slider(
    "Порог уверенности",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05,
)
image_size = st.sidebar.select_slider("Размер изображения", options=[320, 416, 512, 640], value=416)

model = load_model(model_path)

uploaded_file = st.file_uploader(
    "Загрузите изображение",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is None:
    st.info("Загрузите фото строительной или производственной сцены, чтобы запустить детекцию.")
    st.stop()

image_bytes = uploaded_file.getvalue()
prediction_id = create_prediction_id(image_bytes, confidence, image_size)
image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
detections = predict_image(model, image, confidence, image_size)
annotated_image = draw_detections(image, detections)
table = detections_to_frame(detections)
artifact_dir = save_prediction_artifacts(
    prediction_id=prediction_id,
    original_image=image,
    annotated_image=annotated_image,
    detections=detections,
    source_filename=uploaded_file.name,
    confidence=confidence,
    image_size=image_size,
)

helmet_count = sum(item["class_name"] == "helmet" for item in detections)
vest_count = sum(item["class_name"] == "vest" for item in detections)
head_count = sum(item["class_name"] == "head" for item in detections)

class_labels_ru = {
    "helmet": "каска",
    "vest": "жилет",
    "head": "голова",
    "person": "человек",
}

display_table = table.copy()
if not display_table.empty:
    display_table["class"] = display_table["class"].map(class_labels_ru).fillna(display_table["class"])
    display_table = display_table.rename(
        columns={
            "class": "класс",
            "confidence": "уверенность",
        }
    )

if not detections:
    status_title = "Объекты не найдены"
    status_text = "Попробуйте снизить порог уверенности или загрузить другое изображение."
    status_kind = "info"
elif head_count > helmet_count:
    status_title = "Есть риск нарушения"
    status_text = "Модель нашла больше голов, чем касок. Стоит вручную проверить изображение."
    status_kind = "warning"
else:
    status_title = "СИЗ обнаружены"
    status_text = "Модель нашла каски и/или жилеты. Проверьте рамки на изображении."
    status_kind = "success"

st.markdown(
    f"""
    <div class="metric-row">
      <div class="metric-box">
        <div class="metric-label">Всего объектов</div>
        <div class="metric-value">{len(detections)}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Каски</div>
        <div class="metric-value">{helmet_count}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Жилеты</div>
        <div class="metric-value">{vest_count}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.4, 1])

with left:
    st.image(annotated_image, use_container_width=True)

with right:
    if status_kind == "warning":
        st.warning(f"{status_title}. {status_text}")
    elif status_kind == "success":
        st.success(f"{status_title}. {status_text}")
    else:
        st.info(f"{status_title}. {status_text}")

    st.dataframe(display_table, use_container_width=True, hide_index=True)
    st.download_button(
        "Скачать результат",
        data=image_to_bytes(annotated_image),
        file_name="ppe_detection.png",
        mime="image/png",
    )

    st.caption(f"ID проверки: `{prediction_id}`")
    st.caption(f"Файлы сохранены: `{artifact_dir}`")

    st.divider()
    st.subheader("Оценка прогноза")
    feedback = st.radio(
        "Результат модели",
        options=["Верно", "Есть ошибка", "Не уверен"],
        horizontal=True,
    )
    comment = st.text_area("Комментарий", placeholder="Например: модель пропустила каску слева")

    if st.button("Сохранить оценку"):
        append_feedback(prediction_id, feedback, comment)
        st.success("Оценка сохранена.")
