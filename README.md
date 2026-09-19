````markdown
# 🌙 Luna-tics

AI-powered lunar image registration and matching system.

Luna-tics processes lunar imagery using preprocessing, illumination handling, RoMaV2 feature matching, geometric verification, and image registration.

## Features

- Lunar image preprocessing
- Illumination-aware processing
- RoMaV2 feature matching
- Geometric verification
- Image registration
- Match-point visualization
- FastAPI backend
- Web-based frontend

## Project Structure

```text
lunatics_demo/
├── backend/
│   ├── api/
│   ├── lunar_registration/
│   ├── models/
│   ├── uploads/
│   ├── outputs/
│   └── requirements.txt
└── frontend/
    ├── index.html
    ├── css/
    └── js/
````

## Model Setup

The fine-tuned RoMaV2 model is not included in this repository because of its large size.

Create the models directory:

```bash
mkdir -p backend/models
```

Place the model at:

```text
backend/models/romav2_stereolunar_finetuned.pt
```

The model is required to run the registration pipeline.

## Backend Setup

```bash
cd backend
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
```

Make sure RoMaV2 is available at:

```text
backend/third_party/RoMaV2/
```

Run the backend:

```bash
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

API:

```text
http://127.0.0.1:8000
```

Health check:

```text
http://127.0.0.1:8000/api/health
```

## Frontend Setup

In another terminal:

```bash
cd frontend
python3 -m http.server 5173
```

Open:

```text
http://127.0.0.1:5173
```

## Registration API

```http
POST /api/register
```

Parameters:

* `source` — source lunar image
* `reference` — reference lunar image
* `sensor` — imaging sensor

Supported sensors:

```text
OHRC
IIRS
TMC
LROC
```

## Runtime Files

The following files and directories are excluded from Git:

```text
backend/models/romav2_stereolunar_finetuned.pt
backend/uploads/
backend/outputs/
backend/.venv/
backend/.venv312/
__pycache__/
*.pyc
.DS_Store
```

`uploads/` and `outputs/` are generated automatically when the application runs.
