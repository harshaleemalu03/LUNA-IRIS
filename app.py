import os
import sys
import torch

import spaces
import uvicorn
from huggingface_hub import hf_hub_download


ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(ROOT, "backend")
ROMA_SRC = os.path.join(BACKEND, "third_party", "RoMaV2", "src")

if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

if ROMA_SRC not in sys.path:
    sys.path.insert(0, ROMA_SRC)


# Download the exact existing RoMa2 model.
MODEL_PATH = hf_hub_download(
    repo_id="harshaleemalu03/LUNA-IRiS-models",
    filename="romav2_stereolunar_finetuned.pt",
    local_dir=os.path.join(ROOT, "models"),
)

os.environ["ROMA2_WEIGHTS_PATH"] = MODEL_PATH


# Import RoMaV2 first so all modules that do
# "from romav2.device import device" are loaded.
import romav2

# Force every loaded RoMaV2 module's copied device reference to CPU.
ROMA_CPU = torch.device("cpu")

for module_name, module in list(sys.modules.items()):
    if module_name == "romav2" or module_name.startswith("romav2."):
        if hasattr(module, "device"):
            module.device = ROMA_CPU


# Now import the backend.
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