import os
from dataclasses import dataclass
from dotenv import load_dotenv

from common.path_utils import get_file_path

load_dotenv(get_file_path(".env"))
load_dotenv()


@dataclass(frozen=True)
class Config:
    """Runtime configuration loaded from environment variables."""

    DEEPSEEK_API_KEY: str | None = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_BASE_URL: str | None = os.getenv("DEEPSEEK_BASE_URL")
    DEEPSEEK_MODEL_NAME: str | None = os.getenv("DEEPSEEK_MODEL_NAME")
    VOICE_MODEL_PATH: str | None = os.getenv("VOICE_MODEL_PATH")
    VOICE_VAD_MODEL_PATH: str | None = os.getenv("VOICE_VAD_MODEL_PATH")
    FFMPEG_PATH: str | None = os.getenv("FFMPEG_PATH")
    FFPROBE_PATH: str | None = os.getenv("FFPROBE_PATH")
    MYSQL_HOST: str | None = os.getenv("MYSQL_HOST")
    MYSQL_USER: str | None = os.getenv("MYSQL_USER")
    MYSQL_PASSWORD: str | None = os.getenv("MYSQL_PASSWORD")
    MYSQL_DATABASE_NAME: str | None = os.getenv("MYSQL_DATABASE_NAME")


if __name__ == "__main__":
    conf = Config()
    print(conf)
