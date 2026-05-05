import streamlit as st
from python_a2a import AgentNetwork, A2AClient, AIAgentRouter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import Config
import json
from datetime import datetime
import pytz
import re  # 用于清理响应
import logging  # 增加日志模块

# 设置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# 设置页面配置
st.set_page_config(page_title="基于A2A的SmartVoyage旅行助手系统", layout="wide", page_icon="🤖")

# 自定义 CSS 打造高端大气科技感，优化对比度
st.markdown("""
<style>
body {
    font-family: 'Arial', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #1e1b4b 0%, #3b3e99 100%); /* 柔和的深紫到蓝紫渐变 */
    color: #f5f5d5; /* 整体文字颜色为浅黄色，温暖且突出 */
}
.stTextInput > div > input {
    background-color: #3a4a77; /* 输入框背景稍浅 */
    color: #f5f5d5; /* 输入框文字为浅黄色 */
    border-radius: 12px;
    border: 1px solid #7a8acc; /* 边框颜色更柔和 */
    padding: 12px;
}
.stChatMessage {
    background-color: #2c3e50; /* 深蓝灰背景，提高对比度 */
    border-radius: 12px;
    padding: 15px;
    margin-bottom: 15px;
    box-shadow: 0 3px 6px rgba(0,0,0,0.2);
    color: #ecf0f1; /* 浅灰白文字，确保易读 */
    font-size: 1.1em;
    line-height: 1.5;
}
.stChatMessage.user {
    background-color: #34495e; /* 用户消息背景稍亮，增强区分 */
    color: #ecf0f1; /* 浅灰白文字 */
    font-size: 1.1em;
    line-height: 1.5;
}
.stExpander {
    background-color: #3a4a77; /* 扩展面板背景稍浅 */
    border-radius: 12px;
    border: 1px solid #7a8acc;
    transition: all 0.3s ease;
}
.stExpander > summary {
    font-size: 1.3em; /* 增大标题文字 */
    font-weight: 500; /* 稍加粗标题，增强可读性 */
    color: #d4d9ff; /* 保持与 h1, h2, h3 相同的颜色 */
}
.stExpander:hover {
    border-color: #8a9cff; /* 悬停边框颜色更柔和 */
    box-shadow: 0 4px 10px rgba(0,0,0,0.25);
}
h1 {
    color: #d4d9ff; /* 标题颜色稍调整为柔和紫色 */
    font-weight: 300;
    font-size: 2.5em;
}
h2, h3 {
    color: #d4d9ff; /* 二级标题颜色同上 */
    font-weight: 300;
}
.card-title {
    color: #8a9cff; /* 卡片标题颜色调整为柔和蓝色 */
    font-size: 1.6em;
    margin-bottom: 10px;
}
.card-content {
    color: #f5f5d5; /* 卡片内容文字为浅黄色 */
    font-size: 0.95em;
}
.footer {
    text-align: center;
    color: #d4d9ff; /* 页脚文字为柔和紫色 */
    padding: 20px;
    font-size: 1.2em;
}
</style>
""", unsafe_allow_html=True)

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_network" not in st.session_state:
    # 初始化网络和路由器
    network = AgentNetwork(name="Travel Assistant Network")
    network.add("Weather Query Assistant", "http://localhost:5005")
    network.add("Ticket Query Assistant", "http://localhost:5006")
    st.session_state.agent_network = network
    st.session_state.router = AIAgentRouter(
        llm_client=A2AClient("http://localhost:6666"),  # 假设使用router_A2Aagent_Server.py的LLM服务器
        agent_network=network
    )
    conf = Config()
    st.session_state.llm = ChatOpenAI(
        model=conf.model_name,
        api_key=conf.api_key,
        base_url=conf.api_url,
        temperature=0
    )
    # 存储代理的 URL 信息
    st.session_state.agent_urls = {
        "Weather Query Assistant": "http://localhost:5005",
        "Ticket Query Assistant": "http://localhost:5006"
    }
    # 存储对话历史用于意图识别
    st.session_state.conversation_history = ""

# 意图识别和槽位提取的 Prompt（专业设计，确保鲁棒性）
intent_prompt = ChatPromptTemplate.from_template(
    """
系统提示：您是一个专业的旅行意图识别专家，基于用户查询和对话历史，识别意图并提取槽位。严格遵守规则：
- 支持意图：['weather' (天气查询), 'flight' (机票查询), 'train' (高铁/火车票查询), 'concert' (演唱会票查询), 'attraction' (景点推荐)] 或其组合（如 ['weather', 'flight']）。
- 如果意图超出范围，返回意图 'out_of_scope'。
- 提取槽位：
  - weather: city (城市，多个用逗号分隔), date (日期，支持'今天'/'明天'/'后天'/'未来X天'，转换为YYYY-MM-DD或范围)。
  - flight/train: departure_city (出发城市), arrival_city (到达城市), date (日期), seat_type (座位类型，如'经济舱'/'硬卧')。
  - concert: city (城市), artist (艺人), date (日期), ticket_type (票务类型，如'看台、VIP')。
  - attraction: city (城市), preferences (偏好，如'历史'/'自然')。
- 如果意图为组合，只提取公共槽位，并在后续处理中分别填充。
- 如果槽位缺失，返回 'missing_slots' 列表和追问消息,不得回复空信息。
- 对于weather：如果无city，默认['北京','上海','广州','深圳']；无date，默认今天。
- 输出严格为JSON：{{"intents": ["intent1", "intent2"], "slots": {{"intent1": {{"slot1": "value1"}}, "intent2": {{"slot2": "value2"}}}}, "missing_slots": {{"intent1": ["slot1"]}}, "follow_up_message": "追问消息"}}。不要添加额外文本！
- 当前日期：{current_date} (Asia/Shanghai)。
- 基于整个对话历史填充槽位，优先最新查询。

对话历史：{conversation_history}
用户查询：{query}
    """
)

# 天气结果总结 Prompt（优化为专业、鲁棒）
summarize_weather_prompt = ChatPromptTemplate.from_template(
    """
系统提示：您是一位专业的天气预报员，以生动、准确的风格总结天气信息。基于查询和结果：
- 核心：城市、日期、温度范围、天气描述、湿度、风向、降水。
- 如果结果为空，提示“未找到数据，请确认城市/日期，不得编造。”
- 语气：专业预报，如“根据最新数据，北京2025-07-31的天气预报为...”。
- 保持中文，100-150字。
- 如果查询无关，返回“请提供天气相关查询。”

查询：{query}
结果：{raw_response}
    """
)

# 票务结果总结 Prompt（优化为专业、鲁棒）
summarize_ticket_prompt = ChatPromptTemplate.from_template(
    """
系统提示：您是一位专业的旅行顾问，以热情、精确的风格总结票务信息。基于查询和结果：
- 核心：出发/到达、时间、类型、价格、剩余座位。
- 如果结果为空，提示“未找到数据，请确认条件，不得编造。”
- 语气：顾问式，如“为您推荐北京到上海的机票选项...”。
- 保持中文，100-150字。
- 如果查询无关，返回“请提供票务相关查询。”

查询：{query}
结果：{raw_response}
    """
)

# 景点推荐 Prompt（直接生成，专业设计）
attraction_prompt = ChatPromptTemplate.from_template(
    """
系统提示：您是一位旅行专家，基于用户查询生成景点推荐。规则：
- 推荐3-5个景点，包含描述、理由、注意事项。
- 基于槽位：城市、偏好。
- 语气：热情推荐，如“推荐您在北京探索故宫...”。
- 备注：内容生成，仅供参考。
- 保持中文，150-250字。

查询：{query}
槽位：{slots}
    """
)

# 主界面布局
st.title("🤖 基于A2A的SmartVoyage旅行智能助手")
st.markdown("欢迎体验智能对话！输入问题，系统将精准识别意图并提供服务。")

# 两栏布局：左侧对话，右侧 Agent Card
col1, col2 = st.columns([2, 1])

# 左侧对话区域
with col1:
    st.subheader("💬 对话")
    # 对话历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 输入框
    if prompt := st.chat_input("请输入您的问题..."):
        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.conversation_history += f"\nUser: {prompt}"

        # 获取 LLM 和当前日期
        llm = st.session_state.llm
        current_date = datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d')

        # 意图识别
        with st.spinner("正在分析您的意图..."):
            try:
                chain = intent_prompt | llm
                intent_response = chain.invoke({"conversation_history": st.session_state.conversation_history, "query": prompt, "current_date": current_date}).content.strip()
                # 增加日志打印
                logger.info(f"意图识别原始响应: {intent_response}")
                # 清理响应，确保是JSON
                intent_response = re.sub(r'^```json\s*|\s*```$', '', intent_response).strip()
                logger.info(f"清理后响应: {intent_response}")
                intent_output = json.loads(intent_response)
                intents = intent_output.get("intents", [])
                slots = intent_output.get("slots", {})
                missing_slots = intent_output.get("missing_slots", {})
                follow_up_message = intent_output.get("follow_up_message", "")
                logger.info(f"解析意图输出: {intent_output}")

                if "out_of_scope" in intents:
                    response = "您好!我是AI旅行助手SmartVoyage，专注于旅行相关的信息和服务，比如火车票、高铁票、演唱会门票、天气查询、景点推荐等。如果您有任何旅行方面的需求，请随时告诉我，我会尽力为您提供帮助!"
                elif missing_slots:
                    response = follow_up_message
                    st.session_state.conversation_history += f"\nAssistant: {response}"
                else:
                    responses = []
                    routed_agents = []
                    for intent in intents:
                        agent_name = "Weather Query Assistant" if intent == "weather" else "Ticket Query Assistant" if intent in ["flight", "train", "concert"] else None
                        if intent == "attraction":
                            # 直接生成景点推荐
                            chain = attraction_prompt | llm
                            rec_response = chain.invoke({"query": prompt, "slots": json.dumps(slots.get(intent, {}), ensure_ascii=False)}).content.strip()
                            responses.append(rec_response)
                        elif agent_name:
                            # 构建查询语句，确保满足服务器要求
                            intent_slots = slots.get(intent, {})
                            if intent == "weather":
                                if not intent_slots.get("city"):
                                    intent_slots["city"] = "北京,上海,广州,深圳"  # 默认四个城市
                                if not intent_slots.get("date"):
                                    intent_slots["date"] = current_date  # 默认今天
                                query_str = f"{intent_slots['city']} {intent_slots['date']}"
                            else:
                                query_str = f"{intent} {intent_slots.get('departure_city', '')} {intent_slots.get('arrival_city', '')} {intent_slots.get('date', current_date)} {intent_slots.get('seat_type', '')}".strip()
                                if intent == "concert":
                                    query_str = f"演唱会 {intent_slots.get('city', '')} {intent_slots.get('artist', '')} {intent_slots.get('date', current_date)} {intent_slots.get('ticket_type', '')}".strip()

                            # 调用代理（同步，确保顺序）
                            agent = st.session_state.agent_network.get_agent(agent_name)
                            raw_response = agent.ask(query_str)
                            logger.info(f"{agent_name} 原始响应: {raw_response}")
                            # 总结
                            if agent_name == "Weather Query Assistant":
                                chain = summarize_weather_prompt | llm
                                sum_response = chain.invoke({"query": query_str, "raw_response": raw_response}).content.strip()
                            else:
                                chain = summarize_ticket_prompt | llm
                                sum_response = chain.invoke({"query": query_str, "raw_response": raw_response}).content.strip()
                            responses.append(sum_response)
                            routed_agents.append(agent_name)
                        else:
                            responses.append("暂不支持此意图。")

                    response = "\n\n".join(responses)
                    if routed_agents:
                        response = f"**路由至：{', '.join(set(routed_agents))}**\n\n" + response
                    st.session_state.conversation_history += f"\nAssistant: {response}"

                # 显示助手消息
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except json.JSONDecodeError as json_err:
                logger.error(f"意图识别JSON解析失败，响应内容: {intent_response}")
                error_message = f"意图识别JSON解析失败：{str(json_err)}。请重试。"
                with st.chat_message("assistant"):
                    st.markdown(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
            except Exception as e:
                logger.error(f"处理异常: {str(e)}")
                error_message = f"处理失败：{str(e)}。请重试。"
                with st.chat_message("assistant"):
                    st.markdown(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

# 右侧 Agent Card 区域
with col2:
    st.subheader("🛠️ AgentCard")
    for agent_name in st.session_state.agent_network.agents.keys():
        agent_card = st.session_state.agent_network.get_agent_card(agent_name)
        agent_url = st.session_state.agent_urls.get(agent_name, "未知地址")
        with st.expander(f"Agent: {agent_name}", expanded=False):
            st.markdown(f"<div class='card-title'>技能</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='card-content'>{agent_card.skills}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='card-title'>描述</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='card-content'>{agent_card.description}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='card-title'>地址</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='card-content'>{agent_url}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='card-title'>状态</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='card-content'>在线</div>", unsafe_allow_html=True)

# 页脚
st.markdown("---")
st.markdown('<div class="footer">Powered by 黑马程序员 | 基于Agent2Agent的旅行助手系统 v2.0</div>', unsafe_allow_html=True)