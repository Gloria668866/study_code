from typing import Any, TypedDict


class InterViewInfoDict(TypedDict):
    name: str
    company: str
    subject: str
    interview_date_str: str


class AgentState(TypedDict, total=False):
    interview_info_dict: InterViewInfoDict
    input_audio_path: str
    split_audio_path_list: list[str]
    voice_text_list: list[str]
    voice_arrange_text: str
    interview_topic_list: list[dict[str, Any]]
    interview_advice: dict[str, Any]
    interview_markdown_text: str
    record_id: int
