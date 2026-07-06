from src.backend.api import app

if __name__ == "__main__":
    import uvicorn
    from src.core.settings import settings

    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
