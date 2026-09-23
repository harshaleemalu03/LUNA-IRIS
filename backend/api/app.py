from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import math
from fastapi.staticfiles import StaticFiles

from pathlib import Path
import shutil
import uuid

from lunar_registration.pipeline import run_pipeline


# --------------------------------------------------
# App
# --------------------------------------------------

app = FastAPI(title="Luna-tics API")


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://luna-iris.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Directories
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app.mount(
    "/outputs",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="outputs",
)


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Luna-tics backend is running"
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok"
    }


# --------------------------------------------------
# Registration
# --------------------------------------------------

def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]

    if isinstance(obj, tuple):
        return [sanitize_for_json(v) for v in obj]

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None

    return obj

@app.post("/api/register")
async def register(
    source: UploadFile = File(...),
    reference: UploadFile = File(...),
    sensor: str = Form(...)
):

    run_id = str(uuid.uuid4())

    run_upload_dir = UPLOAD_DIR / run_id
    run_output_dir = OUTPUT_DIR / run_id

    run_upload_dir.mkdir(parents=True, exist_ok=True)
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # File paths
    # --------------------------------------------------

    source_path = run_upload_dir / source.filename
    reference_path = run_upload_dir / reference.filename

    try:

        # --------------------------------------------------
        # Save source
        # --------------------------------------------------

        with source_path.open("wb") as buffer:
            shutil.copyfileobj(source.file, buffer)

        # --------------------------------------------------
        # Save reference
        # --------------------------------------------------

        with reference_path.open("wb") as buffer:
            shutil.copyfileobj(reference.file, buffer)

        print()
        print("=" * 60)
        print("LUNA-TICS REGISTRATION")
        print("=" * 60)
        print(f"Run ID:       {run_id}")
        print(f"Sensor:       {sensor}")
        print(f"Source:       {source_path}")
        print(f"Reference:    {reference_path}")
        print(f"Output:       {run_output_dir}")
        print("=" * 60)

        # --------------------------------------------------
        # Run Luna-tics pipeline
        # --------------------------------------------------

        from PIL import Image

        source_img = Image.open(source_path)
        reference_img = Image.open(reference_path)

        source_w, source_h = source_img.size
        reference_w, reference_h = reference_img.size

        MAX_PIXELS = 4_000_000

        source_window = None
        reference_window = None

        if source_w * source_h > MAX_PIXELS:
            scale = (MAX_PIXELS / (source_w * source_h)) ** 0.5
            crop_w = max(1, int(source_w * scale))
            crop_h = max(1, int(source_h * scale))
            x = max(0, (source_w - crop_w) // 2)
            y = max(0, (source_h - crop_h) // 2)
            source_window = (x, y, crop_w, crop_h)

        if reference_w * reference_h > MAX_PIXELS:
            scale = (MAX_PIXELS / (reference_w * reference_h)) ** 0.5
            crop_w = max(1, int(reference_w * scale))
            crop_h = max(1, int(reference_h * scale))
            x = max(0, (reference_w - crop_w) // 2)
            y = max(0, (reference_h - crop_h) // 2)
            reference_window = (x, y, crop_w, crop_h)

        print(f"Source size: {source_w}x{source_h}")
        print(f"Reference size: {reference_w}x{reference_h}")
        print(f"Source window: {source_window}")
        print(f"Reference window: {reference_window}")

        summary = run_pipeline(
            source_path=str(source_path),
            reference_path=str(reference_path),
            out_dir=str(run_output_dir),
            source_sensor=sensor,
            matcher="auto",
            device="cpu",
            use_eloftr=True,
            source_window=source_window,
            reference_window=reference_window,
        )

        print("=" * 60)
        print("LUNA-TICS REGISTRATION COMPLETE")
        print("=" * 60)

        source_stem = Path(source.filename).stem
        matches_filename = f"{source_stem}_matches.png"

        matches_path = run_output_dir / matches_filename

        print(f"Matches image: {matches_path}")

        if not matches_path.exists():
            raise FileNotFoundError(
                f"Matches image was not generated: {matches_path}"
            )

        return {
            "success": True,
            "run_id": run_id,
            "status": "success",
            "output_image": f"/outputs/{run_id}/{matches_filename}"
        }

    except Exception as e:

        print()
        print("=" * 60)
        print("LUNA-TICS PIPELINE FAILED")
        print("=" * 60)
        print(str(e))
        print("=" * 60)

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        await source.close()
        await reference.close()

