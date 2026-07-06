from dataclasses import dataclass

from common.config import Config


@dataclass(frozen=True)
class AppConfig:
    backend_base_url: str = "http://127.0.0.1:8001"
    api_host: str = "0.0.0.0"
    api_port: int = 8001


app_config = AppConfig()
config = Config()
