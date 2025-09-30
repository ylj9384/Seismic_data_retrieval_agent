from typing import Dict, List, Any, TypedDict, Annotated, Sequence
import logging
import json
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from .registry import registry
from .state import AgentSupervisorState
from .dialogue_manager import dialogue_manager
from .clarification import generate_clarification_prompt


# 配置日志
logger = logging.getLogger(__name__)

def route_decision(state: AgentSupervisorState, llm) -> str:
    """根据当前状态决定下一步执行哪个Agent"""
    # logger.info(f"Current state: {state}")
    # 如果已经完成，结束执行
    if state["finished"]:
        return "end"
    
    # 如果已经选择了下一个Agent，直接执行
    if state["next_agent"]:
        next_agent = state["next_agent"]
        # 重置next_agent以便下次决策
        state["next_agent"] = ""
        return next_agent
    
    # 从历史消息中提取对话
    query = state["query"]
    context = state["context"]
    
    # 准备代理描述
    agent_descriptions = []
    for agent_id, info in registry.metadata.items():
        capabilities = ", ".join(info.get("capabilities", []))
        keywords = ", ".join(info.get("keywords", [])[:5])  # 限制关键词数量
        agent_descriptions.append(
            f"- {agent_id}: {info.get('description', '无描述')}\n  能力: {capabilities}\n  关键词: {keywords}"
        )
    agent_descriptions_text = "\n".join(agent_descriptions)
    
    # 上下文信息
    context_items = []
    for key, value in context.items():
        if key.endswith(('_file', '_path')) and value:
            context_items.append(f"- {key}: {value}")
        elif key in ["client_selected", "events_fetched", "waveforms_fetched"]:
            context_items.append(f"- {key}: {value}")
    context_info = "\n".join(context_items) if context_items else "无上下文数据"
    
    # 系统消息
    system_message = SystemMessage(content=f"""你是一个地震学多智能体系统的总监。
    你需要分析用户的查询，并决定使用哪个专业Agent来处理，或者决定任务已完成。

    可用的Agent:
    {agent_descriptions_text}

    当前上下文信息:
    {context_info}

    规则:
    1. 如果用户查询涉及获取或下载地震数据，选择"data_retrieval"
    2. 如果用户查询涉及波形分析或震相识别，且上下文中有波形文件，选择"phase_detection"
    3. 如果任务已完成或用户在表达感谢/再见，选择"end"
    4. 如果不确定，选择"data_retrieval"作为默认值

    只回复以下某个选项（不要添加其他文字）:
    - data_retrieval
    - phase_detection
    - end
    """)
    
    # 用户消息
    user_message = HumanMessage(content=f"用户查询: {query}\n请选择处理此查询的最佳Agent或结束任务。")
    
    # 调用LLM进行决策
    response = llm.invoke([system_message, user_message])
    decision = response.content.strip().lower()
    
    # 验证决策
    valid_options = list(registry.metadata.keys()) + ["end"]
    if decision not in valid_options:
        logger.warning(f"LLM返回了无效的决策: {decision}，使用默认值")
        decision = "data_retrieval"  # 默认值
    
    logger.info(f"路由决策: {decision}")
    return decision

def execute_agent(state: AgentSupervisorState, agent_id: str):
    """执行指定的Agent"""
    logger.info(f"执行Agent: {agent_id}")
    
    # 更新当前Agent
    state["current_agent"] = agent_id
    
    try:
        # 获取Agent和状态类
        agent = registry.build_agent(agent_id)
        state_class = registry.get_state_class(agent_id)
        
        # 准备Agent上下文
        agent_context = prepare_agent_context(agent_id, state["query"], state["context"])
        
        # 创建初始状态
        initial_state = state_class(agent_context)
        
        # 执行Agent
        result = agent.invoke(initial_state)
        # logger.info(f"Agent {agent_id} returned: {result}")
        
        # 保存结果
        state["result"] = result
        
        # 更新全局上下文
        update_context(agent_id, result, state["context"])
        
        # # 添加Agent响应到消息历史
        # if "output" in result and result["output"]:
        #     # 如果输出包含特定关键词，标记为已完成
        #     output_lower = result["output"].lower()
        #     if ("已成功获取" in output_lower or "成功下载" in output_lower or 
        #         "完成分析" in output_lower) and not "需要进一步" in output_lower:
        #         state["finished"] = True
        #         logger.info("任务已完成，标记为结束")
            
        #     # 对于数据检索成功后自动转到波形分析的情况
        #     elif ("波形文件保存" in output_lower or "已保存波形" in output_lower) and "phase_detection" in registry.metadata:
        #         state["next_agent"] = "phase_detection"
        #         logger.info("检测到可进行波形分析，切换到phase_detection")
        
        # 添加Agent响应到消息历史
        if "output" in result and result["output"]:
            logger.info(f"Agent响应: {result['output'][:100]}...")
            state["messages"].append(AIMessage(content=result["output"]))
            state["sender"] = "ai"
            
            # 提取输出文本并转换为小写，方便检测关键词
            output_lower = result["output"].lower()
            
            # 根据agent类型进行特定判断
            if agent_id == "data_retrieval":
                # 1. 检测任务完成标志
                completion_keywords = [
                    "已成功获取", "成功下载", "完成分析", "数据获取成功", 
                    "已保存到", "已下载到", "图表已生成", "绘制成功",
                    "成功绘制", "图表路径", "图像已生成", "波形图已绘制",
                    "查询结果", "查询完成", "数据检索完成", "数据已保存"
                ]

                clarification_keywords = ["请补充", "请提供", "缺少", "补全", "补充参数"]
                
                # 2. 检测数据类型
                has_waveform_data = any(kw in output_lower for kw in [
                    "波形数据", "波形图", "波形文件", "mseed", "sac", "波形已", "波形图已", "波形信息", "完整波形"
                ])
                
                has_event_data = any(kw in output_lower for kw in [
                    "地震事件", "地震目录", "震级", "地震数据", "地震发生", "地震列表"
                ])
                
                has_station_data = any(kw in output_lower for kw in [
                    "台站数据", "台站信息", "台站分布", "台站位置"
                ])
                
                # 3. 检测文件和图表生成
                has_file = "数据文件:" in output_lower or "数据文件：" in output_lower or result.get("data_file")
                has_plot = "图表路径:" in output_lower or "图表路径：" in output_lower or result.get("plot_path")
                
                # 4. 检测可能需要的后续操作
                needs_visualization = (
                    (has_waveform_data or has_event_data or has_station_data) and 
                    not has_plot and
                    any(kw in state["query"].lower() for kw in ["可视化", "画图", "绘制", "展示", "显示"])
                )
                
                needs_download = (
                    (has_waveform_data or has_event_data or has_station_data) and 
                    not has_file and
                    any(kw in state["query"].lower() for kw in ["下载", "保存", "导出"])
                )
                
                # 波形分析检测 - 当有波形数据且查询中提到分析/识别时
                needs_phase_analysis = (
                    has_waveform_data and has_file and
                    ("phase_detection" in registry.metadata) and
                    any(kw in state["query"].lower() for kw in ["分析", "识别", "检测", "拾取", "p波", "s波", "震相"])
                )
                
                # 5. 决策逻辑
                # 如果出现需要补充的提示，则直接返回给用户，等待补充
                if any(kw in output_lower for kw in clarification_keywords):
                    state["finished"] = True
                    logger.info("检测到追问，直接返回给用户，等待补充")
                if has_plot and any(kw in output_lower for kw in ["图表路径", "成功绘制", "已生成图像"]):
                    # 绘图已完成，直接标记结束
                    state["finished"] = True
                    logger.info("绘图任务已完成，标记为结束")
                elif needs_phase_analysis:
                    # 优先考虑震相检测需求
                    state["next_agent"] = "phase_detection"
                    logger.info("检测到波形分析需求，切换到phase_detection")
                elif any(kw in output_lower for kw in completion_keywords) and not needs_visualization and not needs_download:
                    # 如果已完成且不需要进一步处理，标记结束
                    state["finished"] = True
                    logger.info("数据检索任务已完成，无需进一步处理")
                else:
                    # 其他情况，让LLM在下一轮路由决策
                    # 记录当前状态信息以辅助决策
                    if has_waveform_data:
                        state["context"]["has_waveform_data"] = True
                    if has_event_data:
                        state["context"]["has_event_data"] = True
                    if has_station_data:
                        state["context"]["has_station_data"] = True
                    if has_plot:  # 添加这行，记录图像已生成
                        state["context"]["has_plot"] = True
            
            # 震相检测agent完成后直接标记为结束
            elif agent_id == "phase_detection":
                completion_keywords = ["震相识别完成", "拾取完成", "分析完成", "检测结果", "已标记p波", "已标记s波"]
                if any(kw in output_lower for kw in completion_keywords):
                    state["finished"] = True
                    logger.info("震相检测已完成，标记为结束")
        
        return state
        
    except Exception as e:
        logger.error(f"执行Agent {agent_id}失败: {e}")
        error_message = f"执行Agent {agent_id}时出错: {str(e)}"
        state["result"] = {"error": error_message, "output": error_message}
        state["messages"].append(AIMessage(content=error_message))
        state["sender"] = "ai"
        return state

# def execute_agent(state: AgentSupervisorState, agent_id: str):
#     logger.info(f"执行Agent: {agent_id}")
#     state["current_agent"] = agent_id

#     try:
#         agent = registry.build_agent(agent_id)
#         state_class = registry.get_state_class(agent_id)
#         agent_context = prepare_agent_context(agent_id, state["query"], state["context"])
#         initial_state = state_class(agent_context)
#         result = agent.invoke(initial_state)
#         state["result"] = result
#         update_context(agent_id, result, state["context"])

#         # === HITL追问逻辑开始 ===
#         if result.get("clarification_needed"):
#             # 记录追问历史
#             missing_params = result.get("missing_params", [])
#             prompt = generate_clarification_prompt(
#                 missing_params, 
#                 context=state["context"], 
#                 tool_name=agent_id
#             )
#             logger.info(f"参数不全，生成追问: {prompt}")
#             # 写入最终响应，主流程应等待用户补充
#             state["final_response"] = prompt
#             state["finished"] = True  # 标记本轮结束，等待用户补充
#             state["messages"].append(AIMessage(content=prompt))
#             state["sender"] = "ai"
#             return state
#         # === HITL追问逻辑结束 ===

#         if "output" in result and result["output"]:
#             logger.info(f"Agent响应: {result['output'][:100]}...")
#             state["messages"].append(AIMessage(content=result["output"]))
#             state["sender"] = "ai"
            
#             # 提取输出文本并转换为小写，方便检测关键词
#             output_lower = result["output"].lower()
            
#             # 根据agent类型进行特定判断
#             if agent_id == "data_retrieval":
#                 # 1. 检测任务完成标志
#                 completion_keywords = [
#                     "已成功获取", "成功下载", "完成分析", "数据获取成功", 
#                     "已保存到", "已下载到", "图表已生成", "绘制成功",
#                     "成功绘制", "图表路径", "图像已生成", "波形图已绘制",
#                     "查询结果", "查询完成", "数据检索完成", "数据已保存"
#                 ]
                
#                 # 2. 检测数据类型
#                 has_waveform_data = any(kw in output_lower for kw in [
#                     "波形数据", "波形图", "波形文件", "mseed", "sac", "波形已", "波形图已", "波形信息", "完整波形"
#                 ])
                
#                 has_event_data = any(kw in output_lower for kw in [
#                     "地震事件", "地震目录", "震级", "地震数据", "地震发生", "地震列表"
#                 ])
                
#                 has_station_data = any(kw in output_lower for kw in [
#                     "台站数据", "台站信息", "台站分布", "台站位置"
#                 ])
                
#                 # 3. 检测文件和图表生成
#                 has_file = "数据文件:" in output_lower or "数据文件：" in output_lower or result.get("data_file")
#                 has_plot = "图表路径:" in output_lower or "图表路径：" in output_lower or result.get("plot_path")
                
#                 # 4. 检测可能需要的后续操作
#                 needs_visualization = (
#                     (has_waveform_data or has_event_data or has_station_data) and 
#                     not has_plot and
#                     any(kw in state["query"].lower() for kw in ["可视化", "画图", "绘制", "展示", "显示"])
#                 )
                
#                 needs_download = (
#                     (has_waveform_data or has_event_data or has_station_data) and 
#                     not has_file and
#                     any(kw in state["query"].lower() for kw in ["下载", "保存", "导出"])
#                 )
                
#                 # 波形分析检测 - 当有波形数据且查询中提到分析/识别时
#                 needs_phase_analysis = (
#                     has_waveform_data and has_file and
#                     ("phase_detection" in registry.metadata) and
#                     any(kw in state["query"].lower() for kw in ["分析", "识别", "检测", "拾取", "p波", "s波", "震相"])
#                 )
                
#                 # 5. 决策逻辑
#                 if has_plot and any(kw in output_lower for kw in ["图表路径", "成功绘制", "已生成图像"]):
#                     # 绘图已完成，直接标记结束
#                     state["finished"] = True
#                     logger.info("绘图任务已完成，标记为结束")
#                 elif needs_phase_analysis:
#                     # 优先考虑震相检测需求
#                     state["next_agent"] = "phase_detection"
#                     logger.info("检测到波形分析需求，切换到phase_detection")
#                 elif any(kw in output_lower for kw in completion_keywords) and not needs_visualization and not needs_download:
#                     # 如果已完成且不需要进一步处理，标记结束
#                     state["finished"] = True
#                     logger.info("数据检索任务已完成，无需进一步处理")
#                 else:
#                     # 其他情况，让LLM在下一轮路由决策
#                     # 记录当前状态信息以辅助决策
#                     if has_waveform_data:
#                         state["context"]["has_waveform_data"] = True
#                     if has_event_data:
#                         state["context"]["has_event_data"] = True
#                     if has_station_data:
#                         state["context"]["has_station_data"] = True
#                     if has_plot:  # 添加这行，记录图像已生成
#                         state["context"]["has_plot"] = True
            
#             # 震相检测agent完成后直接标记为结束
#             elif agent_id == "phase_detection":
#                 completion_keywords = ["震相识别完成", "拾取完成", "分析完成", "检测结果", "已标记p波", "已标记s波"]
#                 if any(kw in output_lower for kw in completion_keywords):
#                     state["finished"] = True
#                     logger.info("震相检测已完成，标记为结束")
#         return state

#     except Exception as e:
#         logger.error(f"执行Agent {agent_id}失败: {e}")
#         error_message = f"执行Agent {agent_id}时出错: {str(e)}"
#         state["result"] = {"error": error_message, "output": error_message}
#         state["messages"].append(AIMessage(content=error_message))
#         state["sender"] = "ai"
#         return state

def prepare_agent_context(agent_id: str, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """准备Agent执行上下文"""
    # 基础状态
    agent_context = {
        "user_input": query,
        "action": None,
        "action_input": None,
        "tool_results": None,
        "output": None
    }
    
    # 添加历史记录
    history_key = f"{agent_id}_history"
    if history_key in context:
        agent_context["history"] = context[history_key]
    else:
        agent_context["history"] = []
    
    # 根据Agent类型添加特定上下文
    if agent_id == "data_retrieval":
        agent_context.update({
            "client_selected": context.get("client_selected", False),
            "events_fetched": context.get("events_fetched", False),
            "waveforms_fetched": context.get("waveforms_fetched", False)
        })
    elif agent_id == "phase_detection":
        # 尝试从上下文中找到波形文件
        waveform_file = None
        for key in ["waveform_file", "data_file", "last_data_file"]:
            if key in context and context[key]:
                waveform_file = context[key]
                break
        
        if waveform_file:
            agent_context["waveform_file"] = waveform_file
            
            # 修改用户输入，直接包含文件路径
            if "波形文件" not in query and waveform_file not in query:
                agent_context["user_input"] = f" {query}。波形文件已提供：{waveform_file}"
            
            # 添加系统提示
            agent_context["history"].append({
                "role": "system", 
                "content": f"波形文件已提供: {waveform_file}"
            })
            
            # logger.info(f"为phase_detection添加波形文件: {waveform_file}")
    
    
    return agent_context

def update_context(agent_id: str, result: Dict[str, Any], context: Dict[str, Any]):
    """更新全局上下文"""
    # 保存Agent历史
    if "history" in result:
        history_key = f"{agent_id}_history"
        if history_key not in context:
            context[history_key] = []
        context[history_key].extend(result["history"])
    
    # 保存重要字段
    for key in ["data_file", "plot_path", "detection_id", "output"]:
        if key in result and result[key] is not None:
            context[f"last_{key}"] = result[key]
            # 特殊处理波形文件
            if key == "data_file" and agent_id == "data_retrieval":
                context["waveform_file"] = result[key]
    
    # Agent特定字段
    if agent_id == "data_retrieval":
        for key in ["client_selected", "events_fetched", "waveforms_fetched"]:
            if key in result:
                context[key] = result[key]

def finalize_response(state: AgentSupervisorState, llm) -> AgentSupervisorState:
    """总结和增强Agent的响应"""
    # 标记为完成
    state["finished"] = True
    
    # 如果已有Agent结果，直接使用
    if state["result"] and "output" in state["result"] and state["result"]["output"]:
        state["final_response"] = state["result"]["output"]
        return state
    
    # 针对无具体Agent输出的情况，使用LLM生成合适回复
    query = state["query"]
    
    # 系统提示
    system_prompt = """你是一个专业的地震学分析助手。
    你的主要功能是帮助用户获取和分析地震数据。
    
    如果用户的消息是问候语，请友好回应，并简单介绍你的能力。
    如果用户的消息是告别或感谢，请礼貌回应。
    如果是其他闲聊，请简短回答，并引导用户询问地震相关问题。
    始终保持专业、友好和简洁。
    """
    
    # 调用LLM生成回复
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ])
    
    state["final_response"] = response.content
    return state

def create_agent_supervisor(llm):
    """创建Agent Supervisor图"""
    # 定义工作流状态图
    workflow = StateGraph(AgentSupervisorState)

    # 添加路由节点
    workflow.add_node("route", lambda state: state)
    
    # 添加路由节点
    workflow.add_conditional_edges(
        "route",
        lambda state: route_decision(state, llm),
        {
            **{agent_id: f"execute_{agent_id}" for agent_id in registry.metadata.keys()},
            "end": "finalize"
        }
    )
    
    # 添加Agent执行节点
    for agent_id in registry.metadata.keys():
        node_id = f"execute_{agent_id}"
        workflow.add_node(node_id, lambda state, agent=agent_id: execute_agent(state, agent))
        # Agent执行后返回路由器
        workflow.add_edge(node_id, "route")
    
    # 添加终结点
    workflow.add_node("finalize", lambda state: finalize_response(state, llm))
    workflow.add_edge("finalize", END)
    
    # 设置入口点
    workflow.set_entry_point("route")
    
    return workflow.compile()

class AgentSupervisorOrchestrator:
    """基于LangGraph的Agent Supervisor编排器"""
    
    def __init__(self, llm):
        self.llm = llm
        self.graph = create_agent_supervisor(llm)
        logger.info("Agent Supervisor初始化完成")
    
    def orchestrate(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        编排执行过程
        
        Args:
            query: 用户查询
            context: 上下文信息，如果为None则创建新上下文
            
        Returns:
            执行结果
        """
        # 确保有上下文
        if context is None:
            context = {}
        
        # 保存查询历史
        if "queries" not in context:
            context["queries"] = []
        context["queries"].append(query)
        
        # 创建初始状态
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "sender": "human",
            "query": query,
            "next_agent": "",
            "context": context,
            "current_agent": "",
            "result": {},
            "final_response": "",
            "finished": False
        }
        
        try:
            # 执行图 - 使用非流式方式执行
            logger.info("开始执行Agent Supervisor图")
            # 直接使用invoke而不是stream
            final_state = self.graph.invoke(initial_state)
            
            # 格式化结果
            return self._format_result(final_state)
            
        except Exception as e:
            logger.error(f"执行过程出错: {e}")
            return {
                "type": "error",
                "message": f"编排执行出错: {str(e)}",
                "query": query
            }
    
    def _format_result(self, state: AgentSupervisorState) -> Dict[str, Any]:
        """格式化执行结果"""
        # 如果状态为空或不完整，返回错误
        if not state:
            return {
                "type": "error",
                "message": "执行结果不完整",
                "output": "处理过程中发生错误，请重试。"
            }
        
        # 基本结果
        formatted = {
            "type": "orchestrator_result",
            "output": state.get("final_response", "无输出"),
            "last_agent": state.get("current_agent", "none")
        }
        
        # 添加文件和图表路径
        if "context" in state and state["context"]:
            for key, value in state["context"].items():
                if key.endswith(("_file", "_path")) and value:
                    formatted[key] = value
        
        # 添加错误信息(如果有)
        if "result" in state and state["result"] and "error" in state["result"]:
            formatted["error"] = state["result"]["error"]
        
        return formatted