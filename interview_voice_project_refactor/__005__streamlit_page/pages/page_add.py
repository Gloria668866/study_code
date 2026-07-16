import json

import requests
import streamlit as st

from src.core.settings import settings


def page_add():
    if st.button("← 返回主界面"):
        st.session_state.update({"page": "page_main"})
        st.rerun()

    st.title("上传与处理面试录音")
    st.write("上传面试语音文件，系统将自动进行转写、要点提取与分析。")

    name = st.text_input("姓名：")
    company = st.text_input("公司名称：")
    subject = st.selectbox("选择学科", ["python大模型人工智能", "java大模型", "新媒体运营"])
    priority_level = st.radio("处理优先级", ["普通", "优先"], horizontal=True)
    interview_date = st.date_input("面试日期", min_value=None, max_value=None)
    interview_date_str = interview_date.strftime("%Y%m%d")
    file = st.file_uploader(
        "上传面试录音",
        type=["mp3", "wav", "flac", "aac", "m4a"],
        help="限制每个文件最大200MB，支持的格式有：MP3, WAV, FLAC, AAC, M4A",
    )

    json_data = {
        "name": name,
        "company": company,
        "subject": subject,
        "interview_date_str": interview_date_str,
        "priority_level": 1 if priority_level == "优先" else 0,
    }

    if file is not None and name and company and subject and interview_date_str:
        is_click_button = st.button("开始分析")
        if is_click_button:
            try:
                files = {
                    "file": (file.name, file.getbuffer(), getattr(file, "type", "application/octet-stream"))
                }
                data = {"json_data_str": json.dumps(json_data, ensure_ascii=False)}
                resp = requests.post(
                    f"{settings.backend_base_url}/add_interview_record",
                    data=data,
                    files=files,
                    timeout=60,
                )
                if resp.ok:
                    payload = resp.json()
                    record_id = payload.get("record_id")
                    st.session_state.update({"page": "page_main", "last_record_id": record_id, "latest_submitted": record_id})
                    st.success(f"已提交，记录ID：{record_id}，正在返回首页查看进度。")
                    st.rerun()
                else:
                    st.error(f"提交失败：{resp.status_code} {resp.text}")
            except Exception as e:
                st.error(f"提交出错：{e}")
    else:
        st.warning("请填写所有字段并上传录音文件。")
