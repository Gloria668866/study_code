import os
import datetime
import random
from pathlib import Path

import requests
from volcengine.visual.VisualService import VisualService

from agent_state import AgentState
from common.config import Config
from common.path_utils import get_file_path

conf = Config()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


def sanitize_title_for_filename(title: str) -> str:
    """生成简洁且稳定的图片文件名。"""
    now = datetime.datetime.now()
    time_str = now.strftime("%Y%m%d%H%M%S")
    safe_title = "".join(ch for ch in title[:8] if ch.isalnum())
    return f"{time_str}{safe_title}.png"


def generate_jimeng_prompt(title: str, content: str, site: str) -> str:
    return (
        f"请生成一张适合小红书封面的竖版高质量图片，主题必须紧扣：{title}。"
        f"地点信息：{site}。"
        f"内容摘要：{content}。"
        f"画面要像小红书高点击封面一样，第一眼就能看出主题，主体明确、构图干净、视觉重心突出，"
        f"适合手机端浏览，具备强封面感和停留感。"
        f"如果主题是旅行，优先表现当地最具代表性的地标、街景、自然景观或打卡氛围；"
        f"如果主题是生活方式，优先表现人物、场景、物件和情绪氛围；"
        f"如果主题是城市/景点，优先表现地标建筑、路线、局部特写或有代入感的视角。"
        f"整体风格要精致、真实、明亮、统一，颜色协调，层次清晰，具有内容运营封面质感。"
        f"不要生成无关的大风景空镜，不要出现任何文字、logo、水印、边框或拼图排版。"
        f"尽量使用单一主体或双主体构图，保证主题识别度高。"
    )


def download_image_from_url(url: str, output_path: str):
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"图片下载成功: {output_path}")
        return output_path
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"下载图片失败: {str(e)}")
    except IOError as e:
        raise RuntimeError(f"保存图片失败: {str(e)}")


def list_local_images():
    picture_dir = Path(get_file_path("picture"))
    if not picture_dir.exists():
        return []
    return sorted(
        [p for p in picture_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.name,
    )


def use_local_image(output_path: str):
    raise RuntimeError("火山引擎图片生成失败，已中止发布")


def generate_image_with_silence(prompt: str, output_path: str):
    try:
        return generate_image(prompt, output_path)
    except Exception:
        raise


def generate_image(prompt: str, output_path: str):
    ak = conf.JIMENG_AK
    sk = conf.JIMENG_SK
    if not ak or not sk:
        raise RuntimeError("未配置火山引擎 AK/SK")

    visual_service = VisualService()
    visual_service.set_ak(ak)
    visual_service.set_sk(sk)

    form = {
        "req_key": "jimeng_high_aes_general_v21_L",
        "prompt": prompt,
        "return_url": True,
    }

    resp = visual_service.cv_process(form)
    image_urls = resp.get('data', {}).get('image_urls', [])
    if image_urls:
        return download_image_from_url(image_urls[0], output_path)
    raise RuntimeError("图像生成失败，无有效图片链接返回")


def xiaohongshu_image_generator(title, content, site):
    prompt = generate_jimeng_prompt(title, content, site)
    os.makedirs(get_file_path("picture"), exist_ok=True)
    file_name = sanitize_title_for_filename(title)
    output_path = os.path.join(get_file_path("picture"), file_name)
    image_path = generate_image(prompt, output_path)
    return image_path


def image_generate_node(state: AgentState):
    """根据标题和内容生成小红书配图；优先火山引擎，失败则使用本地图片。"""
    try:
        print("开始生成小红书图片生成")
        title = state.get('xiaohongshu_tcm_post_title')
        content = state.get('xiaohongshu_tcm_post_content')
        site = state.get('xiaohongshu_tcm_post_site')

        image_path = xiaohongshu_image_generator(title, content, site)

        state['xiaohongshu_tcm_post_image_path_list'] = [image_path]
        print(f"图片生成成功: {image_path}")
        print("完成生成小红书图片生成")
    except Exception as e:
        import traceback
        traceback.print_exc()
    return state


if __name__ == '__main__':
    print(image_generate_node({"xiaohongshu_tcm_post_title": "韩国旅行",
                               "xiaohongshu_tcm_post_content": "韩国旅行",
                               "xiaohongshu_tcm_post_site": "韩国"}))
