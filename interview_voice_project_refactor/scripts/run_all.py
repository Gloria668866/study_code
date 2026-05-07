from multiprocessing import Process

from scripts.generate_demo_audio import main as generate_demo_audio_main
from scripts.run_api import main as run_api_main
from scripts.run_streamlit import main as run_streamlit_main


def main():
    generate_demo_audio_main()

    api_process = Process(target=run_api_main)
    streamlit_process = Process(target=run_streamlit_main)

    api_process.start()
    streamlit_process.start()

    api_process.join()
    streamlit_process.join()


if __name__ == "__main__":
    main()
