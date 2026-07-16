from playwright.sync_api import sync_playwright

import os

from agent_state import AgentState
from common.path_utils import get_file_path

os.makedirs(get_file_path("cookie"), exist_ok=True)


class XiaohongshuUploader:
    COOKIE_PATH = get_file_path("cookie/xiaohongshu_state.json")
    PUBLISH_URL = "https://creator.xiaohongshu.com/publish/publish?from=homepage&target=image&source=official"
    LOGIN_URL = "https://creator.xiaohongshu.com/login"

    def __init__(self, image_path_list, title="", content=""):
        self.image_path_list = image_path_list
        self.title = title
        self.content = content
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None

    def launch(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=False)

        context_kwargs = {
            "permissions": ["geolocation"],
            "geolocation": {"latitude": 31.2304, "longitude": 121.4737},
        }
        cookie_exists = os.path.exists(self.COOKIE_PATH)
        if cookie_exists:
            print("加载已保存的登录状态")
            context_kwargs["storage_state"] = self.COOKIE_PATH
        else:
            print("未检测到登录状态，创建新上下文")

        self.context = self.browser.new_context(**context_kwargs)
        self.page = self.context.new_page()
        self.page.goto(self.PUBLISH_URL)
        self.wait_seconds(2)

        if self.is_login_required():
            print("检测到登录态失效，跳转登录页重新扫码")
            self.page.goto(self.LOGIN_URL)
            input("请完成扫码登录后按回车继续...")
            self.page.goto(self.PUBLISH_URL)
            self.wait_seconds(2)
            self.context.storage_state(path=self.COOKIE_PATH)
            print("登录状态已重新保存")
        elif not cookie_exists:
            input("请手动登录后按回车继续...")
            self.context.storage_state(path=self.COOKIE_PATH)
            print("登录状态已保存")
        self.wait_seconds(1)

    def close(self):
        self.wait_seconds(6)
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()

    def log(self, message):
        print(message)

    def wait_seconds(self, seconds):
        self.log(f"等待 {seconds} 秒")
        self.page.wait_for_timeout(seconds * 1000)

    def is_login_required(self):
        indicators = [
            'text=登录',
            'button:has-text("登录")',
            'text=扫码登录',
            'text=请先登录',
        ]
        for selector in indicators:
            try:
                if self.page.locator(selector).count() > 0:
                    return True
            except Exception:
                pass
        return False

    def handle_popup(self):
        """处理页面弹窗，通过点击空白区域取消弹窗"""
        self.log("检查并处理页面弹窗")
        try:
            self.wait_seconds(2)
            page_width = self.page.viewport_size["width"]
            page_height = self.page.viewport_size["height"]
            x = int(page_width * 0.9)
            y = int(page_height * 0.1)
            self.page.mouse.click(x, y)
            self.log(f"已点击页面坐标 ({x}, {y}) 处理弹窗")
            self.wait_seconds(1)
        except Exception as e:
            print(f"处理弹窗时发生异常: {str(e)}")

    def switch_to_image_post(self):
        """切换到上传图文模式"""
        self.log("切换到上传图文模式")
        try:
            candidates = [
                '.creator-tab:has-text("上传图文")',
                'text=上传图文',
                'button:has-text("上传图文")',
                '.header-tabs >> text=上传图文',
            ]
            for selector in candidates:
                locator = self.page.locator(selector)
                if locator.count() > 0:
                    locator.first.click(force=True)
                    self.wait_seconds(2)
                    self.log("已切换到上传图文模式")
                    return
            self.log("未找到上传图文入口，继续尝试当前页面")
        except Exception as e:
            self.log(f"切换到上传图文模式失败: {str(e)}")

    def switch_to_video_post(self):
        """切换到上传视频选项"""
        self.log("切换到上传视频模式...")
        try:
            # 等待页面加载完成
            # 等待class为header-tabs的元素加载完成
            self.page.wait_for_selector('.header-tabs', timeout=10000)
            
            # 查找包含"上传视频"文本的tab并点击
            video_tab = self.page.locator('.creator-tab').filter(has_text="上传视频").first
            video_tab.click()
            
            self.log("成功切换到上传视频模式")
            self.wait_seconds(2)  # 等待页面切换完成
        except Exception as e:
            self.log(f"切换到上传视频模式失败: {str(e)}")
            raise

    def upload_images(self):
        """上传图片文件"""
        self.log("开始上传图片")
        try:
            self.page.wait_for_load_state("domcontentloaded")
            trigger_candidates = [
                'button:has-text("上传图片")',
                'button:has-text("添加图片")',
                'button:has-text("上传笔记")',
                'text=上传图片',
                'text=添加图片',
                'text=上传图文',
            ]
            for selector in trigger_candidates:
                locator = self.page.locator(selector)
                if locator.count() > 0:
                    try:
                        locator.first.click(force=True)
                        self.wait_seconds(1)
                        break
                    except Exception:
                        pass

            candidates = [
                'input[type="file"]',
                'input[accept*="image"]',
                '.upload-input',
                'input',
            ]
            file_input = None
            for selector in candidates:
                locator = self.page.locator(selector)
                if locator.count() > 0:
                    for i in range(min(locator.count(), 5)):
                        item = locator.nth(i)
                        try:
                            if item.is_visible() or selector == 'input[type="file"]':
                                file_input = item
                                break
                        except Exception:
                            continue
                if file_input is not None:
                    break

            if file_input is None:
                raise RuntimeError("未找到图片上传输入框")

            try:
                file_input.set_input_files(self.image_path_list)
            except Exception:
                self.page.set_input_files(self.image_path_list)

            self.log(f"已上传 {len(self.image_path_list)} 张图片")
            self.wait_seconds(5)
            self.log("图片上传完成")

        except Exception as e:
            print(f"[!] 图片上传失败: {str(e)}")
            raise

    def fill_title(self):
        """填写标题"""
        self.log("开始填写标题")
        try:
            candidates = [
                'input[placeholder*="标题"]',
                'textarea[placeholder*="标题"]',
                'input[placeholder*="写标题"]',
                '.d-text[placeholder*="填写标题"]',
                'input[type="text"]',
            ]
            title_input = None
            for selector in candidates:
                locator = self.page.locator(selector)
                if locator.count() > 0:
                    title_input = locator.first
                    break

            if title_input is None:
                raise RuntimeError("未找到标题输入框")

            title_input.fill(self.title)
            self.log(f"已填写标题: {self.title}")
            self.wait_seconds(1)
            self.log("标题填写完成")

        except Exception as e:
            print(f"[!] 标题填写失败: {str(e)}")
            raise

    def fill_content(self):
        """填写正文内容"""
        self.log("开始填写正文内容")
        try:
            candidates = ['.ProseMirror', 'div[contenteditable="true"]', 'textarea']
            content_editor = None
            for selector in candidates:
                locator = self.page.locator(selector)
                if locator.count() > 0:
                    content_editor = locator.first
                    break

            if content_editor is None:
                raise RuntimeError("未找到正文编辑器")

            content_editor.click(force=True)
            content_editor.press('Control+a')
            self.wait_seconds(0.5)
            content_editor.type(self.content, delay=50)
            self.log(f"已填写正文内容: {self.content[:50]}...")
            self.wait_seconds(1)
            self.log("正文内容填写完成")

        except Exception as e:
            print(f"[!] 正文内容填写失败: {str(e)}")
            raise

    def submit_post(self):
        """点击发布按钮"""
        self.wait_seconds(5)
        self.log("开始发布小红书")
        try:
            candidates = ['.publishBtn', 'button:has-text("发布")', 'button:has-text("发表")', 'button:has-text("下一步")']
            publish_button = None
            for selector in candidates:
                locator = self.page.locator(selector)
                if locator.count() > 0:
                    publish_button = locator.first
                    break

            if publish_button is None:
                raise RuntimeError("未找到发布按钮")

            if not publish_button.is_enabled():
                self.log("发布按钮不可点击，等待重试")
                self.wait_seconds(3)
                if not publish_button.is_enabled():
                    raise RuntimeError("发布按钮仍然不可点击，发布失败")

            publish_button.click(force=True)
            self.log("已点击发布按钮")
            self.wait_seconds(5)
            self.log("小红书发布完成")

        except Exception as e:
            print(f"[!] 小红书发布失败: {str(e)}")
            raise


def auto_publish_xiaohongshu(image_path_list, title, content):
    print("开始上传小红书")
    xhs = XiaohongshuUploader(image_path_list, title, content)
    xhs.launch()
    xhs.handle_popup()
    xhs.switch_to_image_post()
    xhs.upload_images()
    xhs.fill_title()
    xhs.fill_content()
    xhs.submit_post()
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
