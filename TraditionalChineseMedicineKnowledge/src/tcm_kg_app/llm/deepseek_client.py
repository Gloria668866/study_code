from openai import OpenAI

from tcm_kg_app.config import get_settings


class DeepSeekClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        if not self.settings.model_api_key:
            self.client: OpenAI | None = None
        else:
            self.client = OpenAI(
                api_key=self.settings.model_api_key,
                base_url=self.settings.model_base_url,
            )

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        if self.client is None:
            return ""

        completion = self.client.chat.completions.create(
            model=self.settings.model_name,
            messages=messages,
            temperature=temperature,
        )
        content = completion.choices[0].message.content
        return content or ""


def build_medical_safety_notice() -> str:
    return (
        "本回答仅用于中医药知识查询与技术演示，不构成诊断、处方或治疗建议。"
        "如有严重症状、持续不适、儿童/孕产妇/老人/慢病患者用药问题，请及时咨询专业医生。"
    )
