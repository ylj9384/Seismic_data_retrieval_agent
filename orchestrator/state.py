from typing import Dict, List, Any, TypedDict, Annotated, Sequence
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage

# 定义状态类型
class AgentSupervisorState(TypedDict):
    """Agent Supervisor状态"""
    messages: Annotated[Sequence[BaseMessage], "对话历史"]
    sender: str  # 最后一条消息的发送者
    query: str  # 用户原始查询
    next_agent: str  # 下一步要执行的Agent
    context: Dict[str, Any]  # 执行上下文
    current_agent: str  # 当前执行的Agent
    result: Dict[str, Any]  # 当前执行结果
    final_response: str  # 最终响应
    finished: bool  # 是否完成执行