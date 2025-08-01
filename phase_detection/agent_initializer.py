# filepath: d:\LLM\LangChain\data_retrieval_agent\phase_detection\agent_initializer.py
from langgraph.graph import StateGraph, END, START
from .state import PhaseDetectionState
from .nodes import llm_node, create_tool_node, output_node
from .tool_registry import get_tools

def get_graph():
    """构建震相拾取与事件检测 LangGraph Agent"""
    # 获取工具
    tools = get_tools()
    
    # 初始化状态图
    workflow = StateGraph(PhaseDetectionState)
    
    # 添加LLM节点
    workflow.add_node("llm", llm_node)
    
    # 添加工具节点
    for name, func in tools.items():
        workflow.add_node(name, create_tool_node(name, func))
    
    # 添加输出节点
    workflow.add_node("format_output", output_node)

    # 添加入口点 - 从START到llm节点
    workflow.add_edge(START, "llm") 
    
    # 设置边: LLM → 工具/输出
    workflow.add_conditional_edges(
        "llm",
        lambda state: state["action"],
        {**{name: name for name in tools}, "Final Answer": "format_output"}
    )
    
    # 所有工具节点连接到LLM，以便工具执行完后继续由LLM决定下一步
    for tool_name in tools.keys():
        workflow.add_edge(tool_name, "llm")
    
    # 设置输出节点作为终止节点
    workflow.add_edge("format_output", END)

    return workflow

def build_agent():
    """构建震相拾取与事件检测 LangGraph Agent"""
    return get_graph().compile()

# 确保导出build_agent函数
__all__ = ['get_graph', 'build_agent']