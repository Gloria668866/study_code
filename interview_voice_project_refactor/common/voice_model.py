from functools import lru_cache
from pathlib import Path

from common.config import Config

conf = Config()


class VoiceModelConfigError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_voice_model():
    from funasr import AutoModel

    if not conf.VOICE_MODEL_PATH:
        raise VoiceModelConfigError(
            "缺少环境变量 VOICE_MODEL_PATH，请在 .env 中配置语音识别模型路径或模型名，"
            "例如：VOICE_MODEL_PATH=paraformer-zh"
        )

    model_path = Path(conf.VOICE_MODEL_PATH)
    if model_path.exists():
        model_value = str(model_path)
    else:
        model_value = conf.VOICE_MODEL_PATH

    kwargs = {
        "model": model_value,
        "trust_remote_code": True,
        "vad_kwargs": {"max_single_segment_time": 30000},
        "device": "cpu",
        "disable_update": True,
    }
    if conf.VOICE_VAD_MODEL_PATH:
        vad_path = Path(conf.VOICE_VAD_MODEL_PATH)
        kwargs["vad_model"] = str(vad_path) if vad_path.exists() else conf.VOICE_VAD_MODEL_PATH

    return AutoModel(**kwargs)
