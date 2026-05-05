"""EduRAG Web Server Launcher.
Usage: python run.py
"""
import sys, os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    print(f"Starting EduRAG Web Server at http://{host}:{port}")
    uvicorn.run("web.api:app", host=host, port=port, reload=reload)
