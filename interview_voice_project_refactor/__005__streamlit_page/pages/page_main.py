from datetime import datetime

import requests
import streamlit as st

from src.core.settings import settings

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None


AUTO_REFRESH_INTERVAL_MS = 5000


_BASE_URL = settings.backend_base_url.rstrip("/")

CLEAR_RECORDS_ENDPOINTS = [
    ("DELETE", f"{_BASE_URL}/clear_interview_records"),
    ("POST", f"{_BASE_URL}/clear_interview_records"),
    ("GET", f"{_BASE_URL}/clear_interview_records"),
    ("DELETE", f"{_BASE_URL}/api/clear_interview_records"),
    ("POST", f"{_BASE_URL}/api/clear_interview_records"),
    ("GET", f"{_BASE_URL}/api/clear_interview_records"),
]


def parse_datetime(value):
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return None
    return None


def get_interview_data():
    interview_data = []
    try:
        resp = requests.get(f"{settings.backend_base_url}/interview_records", timeout=30)
        payload = resp.json() if resp.ok else {"data": []}
        raw_list = payload.get("data", []) or []
        for item in raw_list:
            interview_data.append(
                {
                    "id": item.get("id", ""),
                    "name": item.get("name", ""),
                    "interview_time": parse_datetime(item.get("interview_time")),
                    "company_name": item.get("company_name", ""),
                    "processing_status": item.get("processing_status", 0),
                    "priority_level": item.get("priority_level", 0),
                    "processing_tips": item.get("processing_tips", ""),
                    "create_time": parse_datetime(item.get("create_time")),
                    "update_time": parse_datetime(item.get("update_time")),
                }
            )
    except Exception:
        return interview_data, "列表正在刷新中，请稍后查看最新记录。"
    return interview_data, None


def clear_interview_records():
    last_error = None
    for method, url in CLEAR_RECORDS_ENDPOINTS:
        try:
            resp = requests.request(method, url, timeout=10)
            if resp.ok:
                return True, resp.text
            last_error = f"{resp.status_code} {resp.text}"
        except Exception as e:
            last_error = str(e)
    return False, last_error or "未知错误"


def goto_detail_page(record_id):
    st.session_state.update({"record_id": record_id, "page": "page_detail"})
    st.rerun()


def status_text(status):
    if status == 0:
        return "未处理"
    if status == 1:
        return "处理中"
    if status == 2:
        return "已完成处理"
    if status == -1:
        return "处理失败"
    return str(status)


def priority_label(priority_level):
    return "优先" if int(priority_level or 0) == 1 else "普通"


def progress_value(status):
    if status == 0:
        return 0
    if status == 1:
        return 60
    if status == 2:
        return 100
    if status == -1:
        return 100
    return 0


def progress_label(status):
    if status == 0:
        return "等待开始"
    if status == 1:
        return "处理中"
    if status == 2:
        return "处理完成"
    if status == -1:
        return "处理失败"
    return "未知状态"


def page_main():
    st.title("面试录音分析")
    st.write("这是面试录音分析")

    st.session_state.setdefault("page_main_refresh_enabled", True)

    action_col_1, action_col_2, action_col_3 = st.columns([6, 2, 2])
    with action_col_2:
        if st.button("处理语音", use_container_width=True):
            st.session_state.update({"page": "page_add"})
            st.rerun()
    with action_col_3:
        if st.button("清除记录", use_container_width=True):
            ok, message = clear_interview_records()
            if ok:
                st.success("已清除所有记录")
                st.rerun()
            else:
                st.info(f"清除失败：{message}")

    interview_data, fetch_warning = get_interview_data()

    if fetch_warning:
        st.info(fetch_warning)

    if st_autorefresh and any(item["processing_status"] == 1 for item in interview_data):
        st_autorefresh(interval=AUTO_REFRESH_INTERVAL_MS, key="page_main_autorefresh")

    refresh_enabled = st.session_state.get("page_main_refresh_enabled", True)
    st.caption("说明：未处理 / 处理中 / 已完成 / 失败 的状态会随后端任务更新自动刷新。")
    if refresh_enabled and st_autorefresh:
        st.caption("当前页面会在检测到处理中任务时自动刷新。")
    elif refresh_enabled:
        st.caption("提示：安装 `streamlit-autorefresh` 后可在有任务处理时自动刷新。")
    if not interview_data:
        st.info("当前还没有处理记录，上传音频后会自动开始分析。")
    else:
        running_count = sum(1 for item in interview_data if item["processing_status"] == 1)
        failed_count = sum(1 for item in interview_data if item["processing_status"] == -1)
        done_count = sum(1 for item in interview_data if item["processing_status"] == 2)
        waiting_count = sum(1 for item in interview_data if item["processing_status"] == 0)
        st.metric("未处理", waiting_count)
        st.metric("处理中", running_count)
        st.metric("已完成", done_count)
        st.metric("失败", failed_count)

    st.markdown(
        """
        <style>
            .table-header { background-color: #f1f1f1; font-weight: bold; padding: 8px; text-align: left; }
            .table-row-even { background-color: #f9f9f9; padding: 8px; }
            .table-row-odd { background-color: #e9e9e9; padding: 8px; }
            .status-badge { padding: 4px 10px; border-radius: 999px; display: inline-block; font-size: 12px; }
            .status-0 { background: #eef2ff; color: #3730a3; }
            .status-1 { background: #fef3c7; color: #92400e; }
            .status-2 { background: #dcfce7; color: #166534; }
            .status--1 { background: #fee2e2; color: #991b1b; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col_1, col_2, col_3, col_4, col_5, col_6, col_7, col_8 = st.columns([1, 1, 1, 1, 1, 1, 1, 2])
    with col_1:
        st.write("姓名")
    with col_2:
        st.write("面试时间")
    with col_3:
        st.write("公司名称")
    with col_4:
        st.write("优先级")
    with col_5:
        st.write("处理状态")
    with col_6:
        st.write("状态提示")
    with col_7:
        st.write("更新时间")
    with col_8:
        st.write("操作")

    ordered_data = sorted(
        interview_data,
        key=lambda item: (item.get("priority_level", 0), item.get("update_time") or datetime.min),
        reverse=True,
    )

    for i, interview in enumerate(ordered_data):
        with st.container():
            st.progress(progress_value(interview["processing_status"]))
            st.caption(progress_label(interview["processing_status"]))
            cols = st.columns([1, 1, 1, 1, 1, 1, 1, 2])
            with cols[0]:
                st.write(interview["name"])
            with cols[1]:
                st.write(interview["interview_time"])
            with cols[2]:
                st.write(interview["company_name"])
            with cols[3]:
                st.markdown(
                    f'<span class="status-badge status-0">{priority_label(interview.get("priority_level", 0))}</span>',
                    unsafe_allow_html=True,
                )
            with cols[4]:
                st.markdown(
                    f'<span class="status-badge status-{interview["processing_status"]}">{status_text(interview["processing_status"])}</span>',
                    unsafe_allow_html=True,
                )
            with cols[5]:
                st.write(interview["processing_tips"] or "等待处理")
            with cols[6]:
                st.write(interview["update_time"])
            with cols[7]:
                if st.button("查看详情", key=f"button_{i}"):
                    if interview["processing_status"] == 2:
                        goto_detail_page(interview["id"])
                    else:
                        st.warning("请等待处理完成")
