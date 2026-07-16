from fastapi import FastAPI
import sys
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from tcm_kg_app.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Traditional Chinese Medicine Knowledge API",
        description="中医药知识图谱增强问答系统 API",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
