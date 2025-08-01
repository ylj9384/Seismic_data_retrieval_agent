from typing import Dict, Any, List, Optional, TypedDict

class AgentState(TypedDict):
    """Agent 状态定义，包含所有上下文信息"""
    # 用户输入和历史
    user_input: str
    history: List[Dict[str, Any]]
    
    # 工具调用相关
    action: Optional[str]
    action_input: Optional[Dict[str, Any]]
    tool_results: Optional[Dict[str, Any]]
    
    # 地震数据相关
    data_file: Optional[str]
    plot_path: Optional[str]
    events_data: Optional[Dict[str, Any]]
    events_fetched: bool

    # 波形数据相关
    waveforms_data: Optional[Dict[str, Any]]
    waveforms_fetched: bool

    # 台站数据相关
    stations_data: Optional[Dict[str, Any]]
    stations_fetched: bool
    
    # 输出相关
    output: Optional[str]
    
    # 状态标记
    client_selected: bool
    error: Optional[str]