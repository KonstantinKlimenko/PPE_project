FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV YOLO_CONFIG_DIR=/app/Ultralytics
ENV MODEL_PATH=/app/models/ppe_yolo11n_baseline.pt

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-app.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.2.2 torchvision==0.17.2
RUN pip install --no-cache-dir -r requirements-app.txt

COPY app ./app
COPY src ./src
COPY models ./models
COPY README.md .

RUN mkdir -p /app/reports /app/Ultralytics

EXPOSE 7860

CMD streamlit run app/streamlit_app.py \
    --server.address=0.0.0.0 \
    --server.port=${PORT} \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
