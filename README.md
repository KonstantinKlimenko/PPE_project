# Construction PPE Detection

Computer vision project for detecting personal protective equipment on construction and industrial site images.

The project goal is to build a full ML pipeline:

1. Load and validate image data.
2. Explore class balance and annotation quality.
3. Train a YOLO object detection model.
4. Evaluate model quality with detection metrics.
5. Serve predictions through a local API and web interface.
6. Package the application with Docker.

## Dataset

Current expected dataset layout:

```text
data/raw/
+-- images/
|   +-- train/
|   +-- val/
|   +-- test/
+-- labels/
    +-- train/
    +-- val/
    +-- test/
```

The YOLO annotation format is one text file per image:

```text
class_id x_center y_center width height
```

All coordinates are normalized to the `[0, 1]` range.

## Classes

```text
0: Helmet
1: Vest
2: Head
3: Person
```

## First Checks

Run dataset validation:

```bash
python src/check_dataset.py
```

If the script reports missing `labels`, download the annotation archive from the dataset page and put it into `data/raw`.

## Environment

The current local environment is `D:\Anaconda\envs\ml`.

Check CUDA:

```bash
D:\Anaconda\envs\ml\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Check core imports in PowerShell:

```powershell
$env:YOLO_CONFIG_DIR="D:\Jupiter project\Project_for_CV\Ultralytics"
D:\Anaconda\envs\ml\python.exe -c "import numpy, torch, ultralytics, cv2; print(numpy.__version__, torch.cuda.is_available(), ultralytics.__version__, cv2.__version__)"
```

Install project dependencies:

```bash
D:\Anaconda\envs\ml\python.exe -m pip install -r requirements.txt
```

On Windows/Anaconda, stop all Jupyter kernels before installing packages.

## EDA

Run dataset analysis:

```bash
D:\Anaconda\envs\ml\python.exe src\dataset_eda.py
```

Outputs are saved to `reports/`.

## Training

Sanity training from architecture only:

```bash
D:\Anaconda\envs\ml\python.exe src\train.py --model yolo11n.yaml --epochs 1 --imgsz 320 --batch 8 --fraction 0.005 --no-amp --name ppe_yolo_sanity_scratch
```

This checks that the data pipeline, CUDA, training loop, validation, and artifact saving work.

For a real portfolio model, use pretrained weights:

```bash
D:\Anaconda\envs\ml\python.exe src\train.py --model yolo11n.pt --epochs 30 --imgsz 640 --batch 8 --fraction 1.0 --name ppe_yolo11n_full
```

## Prediction

Run prediction on one image:

```bash
D:\Anaconda\envs\ml\python.exe src\predict.py --source data\raw\images\test\000009.jpg --conf 0.25
```

Prediction images are saved to `runs/predict/`.

## API

Run a local FastAPI server:

```powershell
cd "D:\Jupiter project\Project_for_CV"
$env:YOLO_CONFIG_DIR="D:\Jupiter project\Project_for_CV\Ultralytics"
D:\Anaconda\envs\ml\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Health check:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health
```

Local API smoke test without starting a separate server:

```powershell
D:\Anaconda\envs\ml\python.exe src\test_api.py
```

## Web App

Run the Streamlit interface:

```powershell
cd "D:\Jupiter project\Project_for_CV"
$env:YOLO_CONFIG_DIR="D:\Jupiter project\Project_for_CV\Ultralytics"
D:\Anaconda\envs\ml\python.exe -m streamlit run app\streamlit_app.py
```

The app opens a clean image upload interface for local model inference.

The current demo UI is in Russian and shows:

- uploaded image with predicted bounding boxes;
- counts for all detections, helmets, and vests;
- a detection table with confidence and box coordinates;
- a warning if the model detects more heads than helmets.
- local prediction history in `reports/predictions/`;
- user feedback in `reports/feedback.csv`.

## Docker

The Docker image contains the application code, Python dependencies, and the baseline model:

```text
models/ppe_yolo11n_baseline.pt
```

Build and run:

```powershell
cd "D:\Jupiter project\Project_for_CV"
docker compose up --build
```

Open the app:

```text
http://localhost:8501
```

Prediction history and feedback are written to the local `reports/` folder through a Docker volume.

Stop the container:

```powershell
docker compose down
```

## Current Baseline

Final YOLO11n baseline:

```bash
D:\Anaconda\envs\ml\python.exe src\train.py --model yolo11n.pt --epochs 10 --imgsz 640 --batch 8 --fraction 1.0 --no-amp --name ppe_yolo11n_full_10ep_640
```

Best validation metrics:

```text
Precision: 0.858
Recall: 0.809
mAP50: 0.863
mAP50-95: 0.513
```

Summarize a run:

```bash
D:\Anaconda\envs\ml\python.exe src\summarize_run.py --run runs\detect\ppe_yolo11n_full_10ep_640
```

## Notes

The raw dataset and training runs are not stored in Git. The repository includes only the application code, dataset configuration, and the final lightweight YOLO model for demo inference.
