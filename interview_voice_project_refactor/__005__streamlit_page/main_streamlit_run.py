import os

from common.path_utils import get_file_path

streamlit_path = get_file_path("__005__streamlit_page/main_streamlit.py")
print(streamlit_path)
os.system(f"streamlit run {streamlit_path} --server.address=0.0.0.0")