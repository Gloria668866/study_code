from pathlib import Path
import subprocess

from common.path_utils import get_file_path


DEMO_AUDIO_PATH = Path(get_file_path("demo_audio.wav"))

# 用于面试演示的中文测试文案：包含自我介绍、项目经历、技术栈、闭环描述
DEMO_TEXT = (
 """ 面试官您好，我叫李明，毕业于某某大学计算机专业。我的技术方向集中在 Python 后端开发与 AI 应用结合，熟练使用 FastAPI、MySQL、Docker，同时也能使用 LangGraph 来编排复杂的大模型工作流。
我想重点聊一下我做过的面试录音分析项目。这个项目的出发点是帮助求职者通过复盘面试过程来发现自己的表现细节。技术实现上，用户上传面试录音后，后端先进行音频格式转换与预处理，然后调用语音识别服务将音频转为文本。接着，我使用 LangGraph 设计了多个大模型节点：第一个节点用于识别面试官的问题，第二个节点抽取求职者的回答，第三个节点对回答质量做简单评估。这些节点按顺序执行，最终把结构化信息汇总成一份 Markdown 格式的复盘报告。
我在这项目中负责了后端接口开发、数据库设计、文件存储和服务之间的业务编排，确保了整个流程从上传到报告生成高效且稳定。通过这个项目，我不仅加深了对大模型 API 调用的理解，也积累了处理长耗时任务和异步流程的经验。"""
)


def main() -> None:
    DEMO_AUDIO_PATH.parent.mkdir(parents=True, exist_ok=True)

    if DEMO_AUDIO_PATH.exists() and DEMO_AUDIO_PATH.stat().st_size > 0:
        print(f"skip existing: {DEMO_AUDIO_PATH}")
        return

    # 使用 Windows 自带的 SAPI 生成中文 wav，适合本地演示和测试
    ps_script = rf"""
Add-Type -AssemblyName System.Speech
$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer
$speak.SetOutputToWaveFile('{DEMO_AUDIO_PATH.as_posix()}')
$speak.Rate = -1
$speak.Volume = 100
$speak.Speak('{DEMO_TEXT}')
$speak.Dispose()
"""

    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
        check=True,
    )
    print(f"generated: {DEMO_AUDIO_PATH}")


if __name__ == "__main__":
    main()
