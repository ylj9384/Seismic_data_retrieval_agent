from langgraph.graph import StateGraph, END, START
from .state import AgentState
from .nodes import llm_node, create_tool_node, output_node
from .tool_registry import get_tools

# GetEvents 结果可以选择去 PlotCatalog 或 DownloadCatalog
def handle_get_events(state):
    state["events_fetched"] = True
    # 让 LLM 决定下一步操作（绘图还是下载）
    return "llm"
def handle_get_waveforms(state):
    state["waveforms_fetched"] = True
    # 让 LLM 决定下一步操作
    return "llm"

def handle_get_stations(state):
    state["stations_fetched"] = True
    # 让 LLM 决定下一步操作
    return "llm"

def get_graph():
    """构建 LangGraph Agent"""
    # 获取工具
    tools = get_tools()
    
    # 初始化状态图
    workflow = StateGraph(AgentState)
    
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
    
    workflow.add_conditional_edges(
        "GetEvents",
        handle_get_events
    )

    # 为 GetWaveforms 添加条件边
    workflow.add_conditional_edges(
        "GetWaveforms",
        handle_get_waveforms
    )

    # 为 GetStations 添加条件边
    workflow.add_conditional_edges(
        "GetStations",
        handle_get_stations
    )
    
    # Plot 和 Download 结束后直接输出
    workflow.add_edge("PlotCatalog", "format_output")
    workflow.add_edge("DownloadCatalog", "format_output")
    workflow.add_edge("PlotWaveforms", "format_output")
    workflow.add_edge("DownloadWaveforms", "format_output")
    workflow.add_edge("PlotStations", "format_output")
    workflow.add_edge("DownloadStations", "format_output")
    

    # 其他工具节点连接到LLM
    for tool_name in tools.keys():
        if tool_name not in ["GetEvents", "GetWaveforms", "GetStations", 
                             "PlotCatalog", "DownloadCatalog", 
                             "PlotWaveforms", "DownloadWaveforms",
                             "PlotStations", "DownloadStations"]:
            workflow.add_edge(tool_name, "llm")
    
    
    
    # 设置输出节点作为终止节点
    workflow.add_edge("format_output", END)

    return workflow

def build_agent():
    """构建 LangGraph Agent"""
    return get_graph().compile()