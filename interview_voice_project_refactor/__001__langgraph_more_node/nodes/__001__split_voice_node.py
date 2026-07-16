import os
import shutil
import time

from __001__langgraph_more_node.agent_state import AgentState
from __003__fastapi.update_mysql import update_mysql
from common.config import Config
from common.path_utils import get_file_path

conf = Config()

try:
    from pydub import AudioSegment
except Exception:  # pragma: no cover
    AudioSegment = None


async def split_audio(state: AgentState, file_path, output_root="output", segment_length=60000, overlap=5000):
    """
    切割音频文件并返回生成的片段文件路径列表

    :param file_path: 输入的音频文件路径
    :param output_root: 输出的根目录（默认 "output"）
    :param segment_length: 每段音频的时长（毫秒）
    :param overlap: 重叠时长（毫秒）
    :return: 切割后生成的文件路径列表
    """
    await update_mysql("开始切分语音", record_id=state["record_id"], processing_status=1)

    # 提取文件名（不带路径和扩展名）
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # 创建输出目录（年月日_时分秒_文件名）
    output_dir = os.path.join(output_root, f"{base_name}")

    # 如果目录已存在 → 清空目录
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # 删除整个目录
    os.makedirs(output_dir, exist_ok=True)

    if AudioSegment is None:
        raise RuntimeError("pydub is not installed, cannot split audio")

    # 加载音频
    print(f"正在加载音频: {file_path}")
    if conf.FFMPEG_PATH:
        ffmpeg_dir = os.path.dirname(conf.FFMPEG_PATH)
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    audio = AudioSegment.from_file(file_path)
    audio_length = len(audio)

    output_files = []  # 存储生成的文件路径

    # 切割逻辑
    start = 0
    counter = 1
    while start < audio_length:
        end = min(start + segment_length, audio_length)
        segment = audio[start:end]

        # 文件名：年月日_时分秒_文件名_segment_001.wav
        out_file = os.path.join(
            output_dir,
            f"{base_name}_segment_{counter:03d}.wav"
        )
        segment.export(out_file, format="wav")
        print(f"导出: {out_file}")

        output_files.append(out_file)

        # 更新下一个开始点（注意重叠部分）
        start = start + segment_length - overlap
        counter += 1

    await update_mysql("完成切分语音", record_id=state["record_id"])
    return output_files


async def split_voice_node(state: AgentState):
    input_audio_path = state['input_audio_path']
    # 获取voice路径
    out_file_path = get_file_path("voice")

    split_audio_path_list = await split_audio(state, input_audio_path, out_file_path)
    await update_mysql(f"语音共切割了{len(split_audio_path_list)}块", record_id=state["record_id"])
    state['split_audio_path_list'] = split_audio_path_list
    return state


if __name__ == '__main__':
    split_voice_node(
        {"input_audio_path": "/Users/duyi/PycharmProjects/interview_voice_project/__001__data/罗培鑫面试.aac"})
