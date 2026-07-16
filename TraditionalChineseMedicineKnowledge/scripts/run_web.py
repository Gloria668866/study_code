import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_APP = PROJECT_ROOT / "src" / "tcm_kg_app" / "web" / "streamlit_app.py"


if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(WEB_APP)], check=True)
