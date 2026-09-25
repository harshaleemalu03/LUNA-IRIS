from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import math
from fastapi.staticfiles import StaticFiles

from pathlib import Path
import shutil
import uuid

from huggingface_hub import hf_hub_download

from lunar_registration.pipeline import run_pipeline, _parse_window
from lunar_registration.preprocessing import SidecarXmlError


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

# Download EfficientLoFTR checkpoint from the HF model repository
ELOFTR_MODEL_PATH = BASE_DIR / "models" / "eloftr_lunar.ckpt"
ELOFTR_MODEL_PATH.parent.mkdir(exist_ok=True)
if not ELOFTR_MODEL_PATH.exists():
    hf_hub_download(
        repo_id="harshaleemalu03/LUNA-IRiS-models",
        filename="eloftr_lunar.ckpt",
        local_dir=str(ELOFTR_MODEL_PATH.parent),
    )

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
    # Optional product XML for the source image (routing incidence, etc.).
    source_xml: Optional[UploadFile] = File(None),
    sensor: str = Form(...),
    source_window: Optional[str] = Form(None),
    reference_window: Optional[str] = Form(None),
    incidence_deg: Optional[float] = Form(None),
):
    # Parse before the main try/except so malformed input is a client error
    # (422), not a 500 wrapped around the ValueError.
    try:
        src_win = _parse_window(source_window)
        ref_win = _parse_window(reference_window)
    except (ValueError, TypeError) as e:
        raise HTTPException(
            status_code=422,
            detail=f"Malformed crop window (expected 'x,y,w,h'): {e}",
        )

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

        # Optional source product XML: kept beside the uploaded TIFF and
        # forwarded by explicit path — uploads land in a fresh per-run dir
        # where sibling auto-discovery can never see the original layout.
        # Path(...).name strips any directory a hostile filename carries.
        source_sidecar_path = None
        if source_xml is not None:
            xml_name = Path(source_xml.filename or "source.xml").name
            source_sidecar_path = run_upload_dir / xml_name
            with source_sidecar_path.open("wb") as buffer:
                shutil.copyfileobj(source_xml.file, buffer)

        print()
        print("=" * 60)
        print("LUNA-TICS REGISTRATION")
        print("=" * 60)
        print(f"Run ID:       {run_id}")
        print(f"Sensor:       {sensor}")
        print(f"Source:       {source_path}")
        print(f"Reference:    {reference_path}")
        if source_sidecar_path is not None:
            print(f"Source XML:   {source_sidecar_path}")
        print(f"Output:       {run_output_dir}")
        print("=" * 60)

        # --------------------------------------------------
        # Run Luna-tics pipeline
        # --------------------------------------------------

        # Size probe: pre-compute center crops for oversized uploads so the
        # pipeline receives explicit windows (origin/main large-TIFF work).
        # The probe must NEVER abort the request: a file PIL cannot identify
        # is reported by run_pipeline itself, and the API tests stub that
        # stage — on any probe failure both windows stay None and the
        # pipeline's own size guard remains the backstop.
        source_window = None
        reference_window = None
        try:
            from PIL import Image

            source_img = Image.open(source_path)
            reference_img = Image.open(reference_path)

            source_w, source_h = source_img.size
            reference_w, reference_h = reference_img.size

            MAX_PIXELS = 4_000_000

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
        except Exception as probe_exc:
            print(f"size probe skipped ({probe_exc}); "
                  "pipeline size guard applies")

        # Window precedence: the client's explicit crop (Form fields,
        # parsed to src_win/ref_win) always wins over the API's own
        # size-probe crop; either may be None (pipeline guard handles it).
        effective_source_window = (
            src_win if src_win is not None else source_window
        )
        effective_reference_window = (
            ref_win if ref_win is not None else reference_window
        )

        summary = run_pipeline(
            source_path=str(source_path),
            reference_path=str(reference_path),
            out_dir=str(run_output_dir),
            source_sensor=sensor,
            source_window=effective_source_window,
            reference_window=effective_reference_window,
            # Optional source_xml upload = the product XML, so sidecar
            # incidence routes exactly like the CLI's sibling discovery.
            # A typed incidence_deg (mirrors CLI --incidence-deg) still
            # wins by pipeline precedence when both are sent; with neither,
            # routing falls back to the logged placeholder. This supersedes
            # origin/main's hardcoded OHRC scalar (87.886779): real
            # metadata now arrives per-upload via the XML instead of a
            # constant that only fits one pair.
            source_sidecar_xml=(
                str(source_sidecar_path) if source_sidecar_path else None
            ),
            manual_incidence_deg=incidence_deg,
            matcher="auto",
            device="cpu",
            use_eloftr=True,
        )

        # Fail-closed: a run that failed verification is a structured client
        # error (422), never a 200 "success". Raised before the generic
        # handler below, which re-raises HTTPExceptions untouched.
        if not summary.get("passed", False):
            print("=" * 60)
            print("LUNA-TICS REGISTRATION FAILED VERIFICATION")
            print(f"Reason: {summary.get('failure_reason')}")
            print("=" * 60)
            raise HTTPException(
                status_code=422,
                detail={
                    "passed": False,
                    "run_id": run_id,
                    "failure_reason": summary.get("failure_reason"),
                    "low_precision": summary.get("subpixel_refine", {}).get("low_precision"),
                    "subpixel_refine_reason": summary.get("subpixel_refine", {}).get("reason"),
                    "orthogonal_gate": summary.get("orthogonal_gate"),
                    "condition_routing": summary.get("condition_routing"),
                    "gsd_scale_prior": summary.get("gsd_scale_prior"),
                    "summary_path": f"/outputs/{run_id}/summary.json",
                },
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
            "output_image": f"/outputs/{run_id}/{matches_filename}",
            # Fail-closed truth fields (pipeline-repair) + rich metrics
            # payload (origin/main production work) — both sides of the
            # merge ship in the success response.
            "passed": True,
            "failure_reason": None,
            "low_precision": summary.get("subpixel_refine", {}).get("low_precision", False),
            "subpixel_refine_reason": summary.get("subpixel_refine", {}).get("reason"),
            "orthogonal_gate_passed": summary.get("orthogonal_gate", {}).get("passed"),
            "metrics": summary.get("metrics", {}),
            "miho_gcps": summary.get("miho_gcps", {}),
            "subpixel_refine": summary.get("subpixel_refine", {}),
            "chosen_scale": summary.get("chosen_scale"),
            "chosen_rotation_deg": summary.get("chosen_rotation_deg"),
            "gsd_scale_prior": summary.get("gsd_scale_prior"),
        }

    except HTTPException:
        # Keep 422 (fail-closed) / window-parse errors intact.
        raise

    except SidecarXmlError as e:
        # The client sent an XML we cannot use (missing / unparseable /
        # untagged). Routing at a placeholder would hide the bad input —
        # surface it as a client error naming the file.
        raise HTTPException(status_code=422, detail=str(e))

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
        if source_xml is not None:
            await source_xml.close()



