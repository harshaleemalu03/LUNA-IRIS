import os
import sys
import uvicorn

ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(ROOT, "backend")

if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from api.app import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
