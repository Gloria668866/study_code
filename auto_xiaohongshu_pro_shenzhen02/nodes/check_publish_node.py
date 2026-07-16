import os

from agent_state import AgentState
from common.path_utils import get_file_path


def check_publish_node(state: AgentState):
    """
    检查是否可以发布小红书内容
    
    检查条件：
    1. 已生成小红书标题 (xiaohongshu_tcm_post_title)
    2. 已生成小红书内容 (xiaohongshu_tcm_post_content)
    3. 已生成小红书地点 (xiaohongshu_tcm_post_site)
    4. 已生成小红书图片路径列表 (xiaohongshu_tcm_post_image_path_list)
    5. 图片文件确实存在于文件系统中
    
    :param state: AgentState对象，包含当前状态信息
    :return: 更新后的AgentState对象，包含is_can_publish_xiaohongshu字段
    """
    print("开始检查是否可以发布小红书")
    
    # 初始化发布状态为False
    state['is_can_publish_xiaohongshu'] = False
    
    try:
        # 检查标题是否存在且不为空
        title = state.get('xiaohongshu_tcm_post_title')
        if not title or not title.strip():
            print("检查失败: 小红书标题为空")
            return state
            
        # 检查内容是否存在且不为空
        content = state.get('xiaohongshu_tcm_post_content')
        if not content or not content.strip():
            print("检查失败: 小红书内容为空")
            return state
            
            
        # 检查图片路径列表是否存在且不为空
        image_path_list = state.get('xiaohongshu_tcm_post_image_path_list')
        if not image_path_list or len(image_path_list) == 0:
            print("检查失败: 小红书图片路径列表为空")
            return state
            
        # 检查每个图片文件是否真实存在
        valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        for image_path in image_path_list:
            if not os.path.exists(image_path):
                print(f"检查失败: 图片文件不存在 - {image_path}")
                return state

            # 检查文件是否为图片（简单的文件扩展名检查）
            file_extension = os.path.splitext(image_path)[1].lower()
            if file_extension not in valid_extensions:
                print(f"检查失败: 文件不是有效的图片格式 - {image_path}")
                return state
        
        # 所有检查通过，可以发布
        state['is_can_publish_xiaohongshu'] = True
        print("检查通过: 可以发布小红书")
        
    except Exception as e:
        print(f"检查过程中发生异常: {str(e)}")
        state['is_can_publish_xiaohongshu'] = False
    
    print("完成检查是否可以发布小红书")
    return state


if __name__ == '__main__':
    # 测试用例1 - 所有条件满足
    test_state_1 = {
        'xiaohongshu_tcm_post_title': '韩国旅行攻略',
        'xiaohongshu_tcm_post_content': '这是我第一次去韩国旅行的经历...',
        'xiaohongshu_tcm_post_site': '韩国',
        'xiaohongshu_tcm_post_image_path_list': [get_file_path('picture/20251206172224韩国旅行.png')]
    }
    
    # 测试用例2 - 标题为空
    test_state_2 = {
        'xiaohongshu_tcm_post_title': '',
        'xiaohongshu_tcm_post_content': '这是我第一次去韩国旅行的经历...',
        'xiaohongshu_tcm_post_site': '韩国',
        'xiaohongshu_tcm_post_image_path_list': ['/Users/test/test.png']
    }
    
    print("测试用例1结果:")
    print(check_publish_node(test_state_1))
    
    print("\n测试用例2结果:")
    print(check_publish_node(test_state_2))
