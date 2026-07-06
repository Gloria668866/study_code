import sys
from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tcm_kg_app.config import get_settings


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run("tcm_kg_app.main:app", host=settings.api_host, port=settings.api_port, reload=False)
