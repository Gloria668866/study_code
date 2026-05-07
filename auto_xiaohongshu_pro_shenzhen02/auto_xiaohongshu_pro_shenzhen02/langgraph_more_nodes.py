from langgraph.constants import START
from langgraph.graph import StateGraph, END

from agent_state import AgentState
from nodes.auto_publish_xiaohongshu_node import auto_publish_xiaohongshu_node
from nodes.check_publish_node import check_publish_node
from nodes.image_generate_node import image_generate_node
from nodes.text_generate_node import text_generate_node


def build_graph():
    graph_builder = StateGraph(AgentState)
    # 添加节点
    graph_builder.add_node(text_generate_node.__name__, text_generate_node)
    graph_builder.add_node(image_generate_node.__name__, image_generate_node)
    graph_builder.add_node(check_publish_node.__name__, check_publish_node)
    graph_builder.add_node(auto_publish_xiaohongshu_node.__name__, auto_publish_xiaohongshu_node)
    # 添加边
    graph_builder.add_edge(START, text_generate_node.__name__)
    graph_builder.add_edge(text_generate_node.__name__, image_generate_node.__name__)
    graph_builder.add_edge(image_generate_node.__name__, check_publish_node.__name__)

    def check_publish_node_condition(state: AgentState):
        if state.get("is_can_publish_xiaohongshu", False):
            return auto_publish_xiaohongshu_node.__name__
        else:
            return END

    # 添加条件边
    graph_builder.add_conditional_edges(check_publish_node.__name__, check_publish_node_condition)
    graph_builder.add_edge(auto_publish_xiaohongshu_node.__name__, END)
    graph = graph_builder.compile()
    return graph


if __name__ == "__main__":
    input = "我想去大理玩儿。"
    graph = build_graph()
    result = graph.invoke({"input": input})
    print(result)
