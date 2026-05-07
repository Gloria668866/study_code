import requests
import streamlit as st

from src.core.settings import settings


STATUS_LABELS = {
    0: "未处理",
    1: "处理中",
    2: "已完成",
    -1: "处理失败",
}


def get_data_by_id(record_id):
    resp = requests.get(
        f"{settings.backend_base_url}/get_interview_records_by_record_id",
        params={"record_id": record_id},
        timeout=60,
    )
    if resp.ok:
        result_dict = resp.json()
        data_dict = result_dict.get("data", {})
        return data_dict
    st.error(f"获取详情失败：{resp.status_code} {resp.text}")
    return {}


def page_detail():
    if st.button("← 返回主界面"):
        st.session_state.update({"page": "page_main"})
        st.rerun()
    st.title("面试评价详情页")

    record_id = st.session_state.get("record_id", "")
    data_dict = get_data_by_id(record_id)
    markdown_text = str(data_dict.get("markdown_text", ""))
    interview_time = str(data_dict.get("interview_time", ""))
    name = str(data_dict.get("name", "未知姓名"))
    company_name = str(data_dict.get("company_name", "未知公司"))
    status = int(data_dict.get("processing_status", 0) or 0)
    priority_level = int(data_dict.get("priority_level", 0) or 0)
    processing_tips = str(data_dict.get("processing_tips", ""))

    st.write(f"优先级：{'优先' if priority_level == 1 else '普通'}")
    st.write(f"状态：{STATUS_LABELS.get(status, status)}")
    if processing_tips:
        st.caption(processing_tips)

    if status == 0:
        st.info("当前记录还未开始处理。")
        return
    if status == 1:
        st.info("当前记录正在处理，请稍后刷新查看结果。")
        return
    if status == -1:
        st.error("当前任务失败，请返回首页查看错误提示后重新提交。")
        return

    if not markdown_text:
        st.warning("任务已完成，但详情内容还未写入，请稍后再试。")
        return

    st.download_button(
        "下载md文件",
        data=markdown_text,
        file_name=f"{interview_time[:10]}_{name}_{company_name}_面试评价.md",
        mime="text/markdown",
    )
    st.markdown(markdown_text)
