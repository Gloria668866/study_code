import os
import socket
import subprocess

import uvicorn

from src.backend.api import app
from src.core.settings import settings


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
    pids = _find_processes_using_port(port)
    for pid in pids:
        if pid != os.getpid():
            _kill_process(pid)


def main():
    ensure_port_available(settings.api_port)
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main()
