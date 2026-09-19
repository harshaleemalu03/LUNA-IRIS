import os
import sys
import spaces
import uvicorn

ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(ROOT, "backend")

if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

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

    uvicorn.run(app, host="0.0.0.0", port=7860)
