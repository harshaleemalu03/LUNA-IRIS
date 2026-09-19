import os
import sys
import importlib
import torch

import spaces
import uvicorn
from huggingface_hub import hf_hub_download


ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(ROOT, "backend")

if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


# Make the bundled RoMaV2 package importable before importing the backend.
ROMA_SRC = os.path.join(BACKEND, "third_party", "RoMaV2", "src")

if ROMA_SRC not in sys.path:
    sys.path.insert(0, ROMA_SRC)


# Download the exact existing RoMa2 model from the public model repository.
MODEL_PATH = hf_hub_download(
    repo_id="harshaleemalu03/LUNA-IRiS-models",
    filename="romav2_stereolunar_finetuned.pt",
    local_dir=os.path.join(ROOT, "models"),
)

os.environ["ROMA2_WEIGHTS_PATH"] = MODEL_PATH


# RoMaV2 has its own global device selector.
# Force that selector to the same CPU device used by the matcher.
try:
    roma_device = importlib.import_module("romav2.device")
    roma_device.device = torch.device("cpu")
except Exception as e:
    print(f"RoMaV2 device setup: {e}", flush=True)


from api.app import app


@spaces.GPU
def gpu_marker():
    return "LUNA-IRiS GPU enabled"


if __name__ == "__main__":
    from spaces.zero import startup

    try:
        startup()
    except Exception as e:
        print(f"ZeroGPU startup: {e}", flush=True)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
    )