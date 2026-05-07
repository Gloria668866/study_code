import os

import requests
import datetime
from volcengine.visual.VisualService import VisualService
from agent_state import AgentState
from common.config import Config
from common.path_utils import get_file_path

conf = Config()


def sanitize_title_for_filename(title: str) -> str:
    """
    将标题字符串清洗成适合作为文件名的格式（去除不合法字符，只保留前max_length个字符）

    :param title: 原始标题
    :param max_length: 截取的最大字符数
    :return: 清洗后的文件名部分
    """
    # 获取现在的时间
    now = datetime.datetime.now()
    # 格式化时间
    time_str = now.strftime("%Y%m%d%H%M%S")
    return time_str + title[:5] + ".png"


def generate_jimeng_prompt(title: str, content: str, site: str) -> str:
    return (
        f"一幅围绕旅行探索主题创作的高质量图像，"
        f"画面展现与标题内容相关的旅行场景，如自然风光、城市街景、异国风情或户外探险，"
        f"构图中可以包含人物、交通工具、建筑或自然景观，"
        f"整体氛围充满冒险、自由与美好假期的感觉，色调明亮或富有层次，"
        f"背景可以是山川、海滩、森林、古城、夜景等，"
        f"表达放松、探索、享受生活的情绪。"
        f"图片描述地址为:{site}。"
        f"图片中不能有任何文字。"
        f"允许写实艺术风格，但需保证画面和谐美观、细节丰富。"
    )


def download_image_from_url(url: str, output_path: str):
    """
    从URL下载图片并保存到指定路径
    
    :param url: 图片URL
    :param output_path: 保存路径
    :return: 保存的文件路径
    :raises: RuntimeError 当下载失败时抛出异常
    """
    try:
        # 发送HTTP GET请求获取图片
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # 检查请求是否成功
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 将图片内容写入文件
        with open(output_path, 'wb') as f:
            f.write(response.content)
            
        print(f"图片下载成功: {output_path}")
        return output_path
        
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"下载图片失败: {str(e)}")
    except IOError as e:
        raise RuntimeError(f"保存图片失败: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"处理图片时发生未知错误: {str(e)}")


def generate_image(prompt: str, output_path: str):
    visual_service = VisualService()
    visual_service.set_ak(conf.JIMENG_AK)  # 替换为你自己的 AK
    visual_service.set_sk(conf.JIMENG_SK)  # 替换为你自己的 SK

    form = {
        "req_key": "jimeng_high_aes_general_v21_L",
        "prompt": prompt,
        "return_url": True
    }

    resp = visual_service.cv_process(form)
    image_urls = resp.get('data', {}).get('image_urls', [])
    if image_urls:
        download_image_from_url(image_urls[0], output_path)
        return output_path
    else:
        raise RuntimeError("图像生成失败，无有效图片链接返回")


def xiaohongshu_image_generator(title, content, site):
    # 生成提示词
    prompt = generate_jimeng_prompt(title, content, site)

    # 建了一个文件夹，用于保存图片
    os.makedirs(get_file_path("picture"), exist_ok=True)
    # 生成文件名
    file_name = sanitize_title_for_filename(title)
    # 拼接路径
    output_path = os.path.join(get_file_path("picture"), file_name)
    # 生成图片
    image_path = generate_image(prompt, output_path)
    return image_path


def image_generate_node(state: AgentState):
    """根据标题和内容生成中医养生风格的小红书配图"""
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
