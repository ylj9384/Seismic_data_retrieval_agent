from typing import Dict, Any, List, Optional, TypedDict

class PhaseDetectionState(TypedDict):
    """震相拾取Agent状态定义"""
    # 用户输入和历史
    user_input: str
    history: List[Dict[str, Any]]
    
    # 工具调用相关
    action: Optional[str]
    action_input: Optional[Dict[str, Any]]
    tool_results: Optional[Dict[str, Any]]
    
    # 震相拾取相关
    detection_id: Optional[str]
    detection_results: Optional[Dict[str, Any]]
    plot_path: Optional[str]
    
    # 输出相关
    output: Optional[str]
    
    # 错误信息
    error: Optional[str]