# syntax=docker/dockerfile:1
# LUNA-IRiS API image: FastAPI app + lunar_registration pipeline.
# Runs: uvicorn api.app:app (WORKDIR backend, so `lunar_registration.*`
# and `api.*` resolve from the CWD exactly like the README dev command).
FROM python:3.12-slim

# WHY: the default PyPI torch wheel is the CUDA build — installing it drags
# in ~4 GB of nvidia-* wheels that do not fit on this disk (~6 GB free).
# The CPU index ships the same pinned versions with a "+cpu" local tag,
# which pip ranks above the plain PyPI build for an identical "==<version>"
# pin, so the requirements stay bit-for-bit reproducible.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

# curl          -> compose healthcheck (GET /api/health)
# libgl1        -> libGL.so.1 for the opencv-python wheel
# libglib2.0-0t64 -> libglib-2.0.so.0 for the opencv-python wheel
#                 (trixie package name; the headless wheel alone needs neither)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        libgl1 \
        libglib2.0-0t64 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies in their own layer so source edits never re-download torch.
COPY requirements.txt ./
RUN pip install --no-cache-dir --extra-index-url "${TORCH_INDEX_URL}" -r requirements.txt

# uploads/ and outputs/ are excluded by .dockerignore; they are created
# here (not by the app) so the named volumes inherit non-root ownership.
COPY backend/ ./backend/
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/backend/uploads /app/backend/outputs \
    && chown -R appuser:appuser /app

USER appuser
WORKDIR /app/backend

# Build-time smoke check: fail the build if a CUDA wheel sneaks back in or
# cv2 cannot import (missing system libs) — both surface later as runtime
# surprises otherwise.
RUN python -c "import cv2, torch; \
assert torch.version.cuda is None, f'not a CPU wheel: {torch.__version__}'; \
print('torch', torch.__version__, '-> CPU-only wheel')"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
