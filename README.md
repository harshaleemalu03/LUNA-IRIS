---
title: LUNA-IRiS
sdk: gradio
app_file: app.py
---
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

## Run with Docker Compose

Prerequisites: Docker with the Compose plugin (`docker compose version`).

```bash
docker compose up --build -d
```

Services and ports:

| Service   | Host port | What it serves |
|-----------|-----------|----------------|
| `api`     | `7860` (override with `API_PORT`) | FastAPI (`uvicorn api.app:app`) — health at `http://localhost:7860/api/health`. 7860 by default because 8000 is frequently taken by other local services; the service itself still listens on 8000 inside the compose network. |
| `frontend`| `5173`    | The UI (nginx). It proxies `/api/` and `/outputs/` to the `api` service, so the browser is same-origin and no CORS configuration is needed. Port 5173 matches the origin already in the API's CORS allowlist and the `python3 -m http.server 5173` dev flow. |

Model weights are **not** in the repository (see *Model Setup*). The stack mounts
a models directory read-only at `/app/models` and points the pipeline at it via
`LUNATICS_MODELS_DIR`:

```bash
mkdir -p models                      # default location, or:
MODELS_DIR=/path/to/weights docker compose up --build -d
```

Without weights the neural matchers cannot start, and the pipeline degrades
honestly to the PWIFT contingency fallback — the stack still answers, and a run
that cannot be verified fails closed with HTTP 422 instead of a fake success.

The UI's API base URL is deployment-configurable through `LUNA_API_BASE` on the
`frontend` service (default: empty string = same-origin `/api` proxy). The
checked-in `frontend/config.js` keeps the previous hosted-backend URL for the
non-Docker deployment.

Reading results: each registration writes to the `luna-iris_api-outputs` volume
(`docker volume` data, surviving restarts) and the API serves it at
`http://localhost:5173/outputs/<run_id>/...` — the response body's
`output_image` / `summary_path` fields give the exact paths.

```bash
curl -fsS localhost:7860/api/health    # {"status":"ok"}   (API_PORT overrides)
curl -fsS localhost:5173/api/health    # same response through the nginx proxy
docker compose logs -f api             # pipeline logs
docker compose down                    # stop (add -v to also drop the volumes)
```

