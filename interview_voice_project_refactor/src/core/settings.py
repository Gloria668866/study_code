from dataclasses import dataclass

from common.config import Config


@dataclass(frozen=True)
class AppSettings:
    backend_base_url: str = "http://127.0.0.1:8001"
    streamlit_page_title: str = "面试录音分析"
    streamlit_page_icon: str = "🎧"
    api_host: str = "0.0.0.0"
    api_port: int = 8001


settings = AppSettings()
config = Config()
