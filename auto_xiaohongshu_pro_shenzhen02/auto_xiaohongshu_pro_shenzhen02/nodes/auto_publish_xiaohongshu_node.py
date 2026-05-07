from playwright.sync_api import sync_playwright

import os

from agent_state import AgentState
from common.path_utils import get_file_path

os.makedirs(get_file_path("cookie"), exist_ok=True)


class XiaohongshuUploader:
    COOKIE_PATH = get_file_path("cookie/xiaohongshu_state.json")
    PUBLISH_URL = "https://creator.xiaohongshu.com/publish/publish?from=homepage&target=image&source=official"

    def __init__(self, image_path_list, title="", content=""):
        self.image_path_list = image_path_list
        self.title = title
        self.content = content
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None

    def launch(self):
        # 启动playwright
        self.playwright = sync_playwright().start()
        # 启动浏览器
        self.browser = self.playwright.chromium.launch(headless=False)

        if os.path.exists(self.COOKIE_PATH):
            print("[√] 加载已保存的登录状态...")
            self.context = self.browser.new_context(
                storage_state=self.COOKIE_PATH,
                permissions=["geolocation"],
                geolocation={"latitude": 31.2304, "longitude": 121.4737}
            )
        else:
            print("[!] 未检测到登录状态，创建新上下文...")
            self.context = self.browser.new_context(
                permissions=["geolocation"],
                geolocation={"latitude": 31.2304, "longitude": 121.4737}
            )

        self.page = self.context.new_page()
        self.page.goto(self.PUBLISH_URL)

        if not os.path.exists(self.COOKIE_PATH):
            input("请手动登录后按回车继续...")
            self.context.storage_state(path=self.COOKIE_PATH)
            print("[√] 登录状态已保存")
        self.wait_seconds(1)

    def close(self):
        # 等待4秒
        self.wait_seconds(6)
        self.browser.close()
        self.playwright.stop()

    def wait_seconds(self, seconds):
        print(f"⏳ 等待 {seconds} 秒...")
        self.page.wait_for_timeout(seconds * 1000)

    def handle_popup(self):
        """处理页面弹窗，通过点击空白区域取消弹窗"""
        print("🔍 检查并处理页面弹窗...")
        try:
            # 等待页面加载完成
            self.wait_seconds(2)
            
            # 点击页面空白区域（通常是页面顶部或侧边）
            # 使用页面坐标点击一个不太可能有元素的空白区域
            page_width = self.page.viewport_size["width"]
            page_height = self.page.viewport_size["height"]
            
            # 点击页面右上角空白区域
            x = int(page_width * 0.9)  # 页面宽度的90%
            y = int(page_height * 0.1)  # 页面高度的10%
            
            self.page.mouse.click(x, y)
            print(f"[√] 已点击页面坐标 ({x}, {y}) 处理弹窗")
            self.wait_seconds(1)  # 等待弹窗消失
            
        except Exception as e:
            print(f"[!] 处理弹窗时发生异常: {str(e)}")
            # 不抛出异常，继续执行后续操作

    def switch_to_video_post(self):
        """切换到上传视频选项"""
        print("📹 切换到上传视频模式...")
        try:
            # 等待页面加载完成
            # 等待class为header-tabs的元素加载完成
            self.page.wait_for_selector('.header-tabs', timeout=10000)
            
            # 查找包含"上传视频"文本的tab并点击
            video_tab = self.page.locator('.creator-tab').filter(has_text="上传视频").first
            video_tab.click()
            
            print("[√] 成功切换到上传视频模式")
            self.wait_seconds(2)  # 等待页面切换完成
        except Exception as e:
            print(f"[!] 切换到上传视频模式失败: {str(e)}")
            raise

    def upload_images(self):
        """上传图片文件"""
        print("📤 开始上传图片...")
        try:
            # 等待上传区域加载完成
            self.page.wait_for_selector('.upload-input', timeout=10000)
            
            # 获取文件输入元素
            file_input = self.page.locator('.upload-input')
            
            # 上传所有图片文件
            file_input.set_input_files(self.image_path_list)
            
            print(f"[√] 已上传 {len(self.image_path_list)} 张图片")
            
            # 等待图片上传完成，可以根据实际情况调整等待时间或添加更精确的等待条件
            self.wait_seconds(5)
            
            # 检查是否有上传成功的标识（可以根据实际情况调整）
            # 这里可以添加检查上传成功的逻辑
            
            print("[√] 图片上传完成")
            
        except Exception as e:
            print(f"[!] 图片上传失败: {str(e)}")
            raise

    def fill_title(self):
        """填写标题"""
        print("✏️ 开始填写标题...")
        try:
            # 等待标题输入框加载完成
            self.page.wait_for_selector('.d-text', timeout=10000)
            
            # 获取标题输入框
            title_input = self.page.locator('.d-text[placeholder*="填写标题"]')
            
            # 清空输入框并填写标题
            title_input.clear()
            title_input.fill(self.title)
            
            print(f"[√] 已填写标题: {self.title}")
            
            # 等待一下确保标题已填写
            self.wait_seconds(1)
            
            print("[√] 标题填写完成")
            
        except Exception as e:
            print(f"[!] 标题填写失败: {str(e)}")
            raise

    def fill_content(self):
        """填写正文内容"""
        print("📝 开始填写正文内容...")
        try:
            # 等待正文编辑器加载完成
            self.page.wait_for_selector('.ProseMirror', timeout=10000)
            
            # 获取正文编辑器
            content_editor = self.page.locator('.ProseMirror')
            
            # 点击编辑器获取焦点
            content_editor.click()
            
            # 清空编辑器内容（如果有）
            content_editor.press('Control+a')
            self.wait_seconds(0.5)
            
            # 填写正文内容
            content_editor.type(self.content, delay=50)  # 添加延迟以模拟真实输入
            
            print(f"[√] 已填写正文内容: {self.content[:50]}...")
            
            # 等待一下确保内容已填写
            self.wait_seconds(1)
            
            print("[√] 正文内容填写完成")
            
        except Exception as e:
            print(f"[!] 正文内容填写失败: {str(e)}")
            raise

    def submit_post(self):
        """点击发布按钮"""
        self.wait_seconds(5)
        print("🚀 开始发布小红书...")
        try:
            # 等待发布按钮加载完成
            self.page.wait_for_selector('.publishBtn', timeout=10000)
            
            # 获取发布按钮
            publish_button = self.page.locator('.publishBtn')
            
            # 检查按钮是否可点击
            if not publish_button.is_enabled():
                print("[!] 发布按钮不可点击，可能缺少必要信息或正在处理中")
                # 等待一段时间再检查
                self.wait_seconds(3)
                if not publish_button.is_enabled():
                    raise RuntimeError("发布按钮仍然不可点击，发布失败")
            
            # 点击发布按钮
            publish_button.click()
            
            print("[√] 已点击发布按钮")
            
            # 等待发布完成，可以根据实际情况调整等待时间或添加更精确的等待条件
            self.wait_seconds(5)
            
            # 检查是否有发布成功的标识（可以根据实际情况调整）
            # 这里可以添加检查发布成功的逻辑，例如检查成功提示或页面跳转
            
            print("[√] 小红书发布完成")
            
        except Exception as e:
            print(f"[!] 小红书发布失败: {str(e)}")
            raise


def auto_publish_xiaohongshu(image_path_list, title, content):
    print("🚀 开始上传小红书...")
    xhs = XiaohongshuUploader(image_path_list, title, content)
    # 小红书启动
    xhs.launch()
    
    # 处理可能出现的弹窗
    xhs.handle_popup()
    
    # 切换到上传视频模式
    # xhs.switch_to_video_post()

    
    
    # 上传图片
    xhs.upload_images()

    xhs.wait_seconds(100000)
    
    # 填写标题
    xhs.fill_title()
    
    # 填写正文内容
    xhs.fill_content()
    
    # 点击发布按钮
    xhs.submit_post()
    
    # xhs.wait_seconds(100000)
    # xhs.switch_to_image_post()
    # xhs.upload_images()
    # xhs.fill_title_and_content()
    # # 最后点击发布
    # xhs.submit_post()
    # 小红书关闭
    xhs.close()


def auto_publish_xiaohongshu_node(state: AgentState):
    """自动发布小红书"""
    image_path_list = state['xiaohongshu_tcm_post_image_path_list']
    title = state['xiaohongshu_tcm_post_title']
    content = state['xiaohongshu_tcm_post_content']
    auto_publish_xiaohongshu(image_path_list, title, content)
    state['output'] = '小红书发布成功'
    return state


if __name__ == '__main__':
    auto_publish_xiaohongshu_node(
        {"xiaohongshu_tcm_post_image_path_list": [get_file_path("picture/20251206172224韩国旅行.png")],
         "xiaohongshu_tcm_post_title": "韩国旅行",
         "xiaohongshu_tcm_post_content": "韩国旅行的内容"})
