import os
import subprocess

import streamlit.web.cli as stcli

from common.path_utils import get_file_path


def _find_processes_using_port(port: int):
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", f"Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess"],
            capture_output=True,
            text=True,
            check=False,
        )
        pids = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.isdigit():
                pids.append(int(line))
        return sorted(set(pids))
    except Exception:
        return []


def _kill_process(pid: int):
    try:
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, text=True, check=False)
    except Exception:
        pass


def ensure_port_available(port: int):
    for pid in _find_processes_using_port(port):
        if pid != os.getpid():
            _kill_process(pid)


def main():
    ensure_port_available(8501)
    streamlit_path = get_file_path("__005__streamlit_page/main_streamlit.py")
    stcli.main(["run", streamlit_path, "--server.address=0.0.0.0"])


if __name__ == "__main__":
    main()
