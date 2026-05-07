import streamlit as st

from src.core.settings import settings
from __005__streamlit_page.pages.page_add import page_add
from __005__streamlit_page.pages.page_detail import page_detail
from __005__streamlit_page.pages.page_main import page_main
from __005__streamlit_page.pages.page_test import page_test

st.set_page_config(
    page_title=settings.streamlit_page_title,
    page_icon=settings.streamlit_page_icon,
    layout="wide",
)

hide_sidebar = """
<style>
    section[data-testid="stSidebar"] {
        display: none;
    }
</style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)


PAGE_MAP = {
    "page_main": page_main,
    "page_detail": page_detail,
    "page_add": page_add,
    "page_test": page_test,
}


def main():
    page = st.session_state.get("page", "page_main")
    PAGE_MAP.get(page, page_test)()


if __name__ == "__main__":
    main()
