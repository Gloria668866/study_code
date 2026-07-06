import streamlit as st

from __005__streamlit_page.main_streamlit import main as streamlit_main


def main():
    st.set_page_config(page_title="面试录音分析", page_icon="🎧", layout="wide")
    streamlit_main()


__all__ = ["main"]
