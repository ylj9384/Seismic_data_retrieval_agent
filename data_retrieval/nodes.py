import logging
import json
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from .state import AgentState
from config.llm import get_kimi_llm, get_qwen_llm
from .prompt_templates import create_system_prompt

logger = logging.getLogger(__name__)

def llm_node(state: AgentState) -> Dict:
    """LLM 节点：处理用户输入，决定下一步行动"""
    # llm = get_kimi_llm()
    llm = get_qwen_llm()
    
    # 构建历史消息
    messages = [SystemMessage(content=create_system_prompt())]

    # 添加状态感知提示
    status_prompt = ""

    # 如果有循环检测迹象，添加特殊提示
    if state.get("client_selected") == True:
        messages.append(SystemMessage(content="""
        注意：客户端已经设置成功，不需要再次设置客户端。
        """))

    # 数据状态提示
    if state.get("events_fetched") == True:
        # 分析用户原始请求，判断是否需要图表或下载
        need_plot = any(kw in state.get("user_input", "").lower() for kw in ["图表", "可视化", "图像", "绘制", "画图", "展示", "显示"])
        need_download = any(kw in state.get("user_input", "").lower() for kw in ["下载", "导出", "保存", "原始数据", "数据文件"])
        
        status_prompt += "\n数据已成功获取。"
        
        if need_plot:
            status_prompt += "用户需要图表展示，请使用PlotCatalog工具。"
        elif need_download:
            status_prompt += "用户需要下载数据，请使用DownloadCatalog工具。"
        else:
            status_prompt += "请提供Final Answer总结地震数据。"
    
    # 添加波形数据状态提示
    if state.get("waveforms_fetched") == True:
        # 分析用户原始请求，判断是否需要图表或下载
        need_plot = any(kw in state.get("user_input", "").lower() for kw in ["图表", "可视化", "图像", "绘制", "画图", "展示", "显示"])
        need_download = any(kw in state.get("user_input", "").lower() for kw in ["下载", "导出", "保存", "原始数据", "数据文件"])
        
        status_prompt += "\n波形数据已成功获取。"
        
        if need_plot:
            status_prompt += "用户需要波形图表展示，请使用PlotWaveforms工具。"
        elif need_download:
            status_prompt += "用户需要下载波形数据，请使用DownloadWaveforms工具。"
        else:
            status_prompt += "请提供Final Answer总结波形数据信息。"
    
    # 添加台站数据状态提示
    if state.get("stations_fetched") == True:
        # 分析用户原始请求，判断是否需要图表或下载
        need_plot = any(kw in state.get("user_input", "").lower() for kw in ["图表", "可视化", "图像", "绘制", "画图", "展示", "显示"])
        need_download = any(kw in state.get("user_input", "").lower() for kw in ["下载", "导出", "保存", "原始数据", "数据文件"])
        
        status_prompt += "\n台站数据已成功获取。"
        
        if need_plot:
            status_prompt += "用户需要台站位置分布图，请使用PlotStations工具。"
        elif need_download:
            status_prompt += "用户需要下载台站数据，请使用DownloadStations工具。"
        else:
            status_prompt += "请提供Final Answer总结台站数据信息。"
    
    # 如果有状态提示，添加到消息中
    if status_prompt:
        messages.append(SystemMessage(content=status_prompt))

    # 添加历史消息，包括系统消息
    for msg in state["history"]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
        elif msg["role"] == "system":
            # 关键改动：确保系统消息也被加入上下文
            messages.append(SystemMessage(content=msg["content"]))
    
    # 添加当前用户输入
    messages.append(HumanMessage(content=state["user_input"]))
    
    # 调用LLM
    response = llm.invoke(messages)
    content = response.content
    
    logger.info(f"LLM响应: {content[:100]}...")
    
    try:
        # 尝试从LLM回答中提取JSON
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0].strip()
        else:
            # 如果没有JSON格式，将其视为普通对话回复
            logger.info("LLM返回普通文本，将作为Final Answer处理")
            new_state = state.copy()
            new_state["action"] = "Final Answer"
            new_state["action_input"] = content  # 使用整个文本作为最终答案
            new_state["history"].append({"role": "assistant", "content": content})
            return new_state
        
        parsed_response = json.loads(json_str)
        action = parsed_response.get("action")
        action_input = parsed_response.get("action_input", {})

        # 创建新状态 - 添加此行以修复错误
        new_state = state.copy()
        new_state["action"] = action
        
        # 根据动作类型处理 action_input
        if action == "Final Answer" and isinstance(action_input, str):
            # 如果是最终答案且是字符串，直接使用
            new_state["action_input"] = action_input
        else:
            # 如果是工具调用，确保是字典类型
            new_state["action_input"] = action_input if isinstance(action_input, dict) else {}
        
        # 将LLM回答添加到历史
        new_state["history"].append({"role": "assistant", "content": content})
        
        return new_state
    except Exception as e:
        # 解析失败，返回错误
        logger.error(f"解析LLM输出失败: {e}")
        new_state = state.copy()
        new_state["error"] = f"解析LLM输出失败: {e}"
        new_state["action"] = "Final Answer"
        new_state["action_input"] = {"error": str(e)}
        return new_state

def create_tool_node(tool_name: str, tool_func):
    """为每个工具创建对应的节点"""
    def node_func(state: AgentState) -> Dict:
        """调用工具并处理结果"""
        new_state = state.copy()
        
        # 提取工具参数
        action_input = state["action_input"]
        logger.info(f"调用工具 {tool_name}: {action_input}")
        
        try:
            # 调用工具
            result = tool_func(**action_input)
            
            # 保存工具结果
            new_state["tool_results"] = result
            
            # 关键修改: 添加工具执行结果到历史记录
            result_message = f"工具 {tool_name} 执行结果: {json.dumps(result, ensure_ascii=False)}"
            new_state["history"].append({"role": "system", "content": result_message})
            
            # 更新状态标记
            if tool_name == "SelectClient" and result.get("status") == "success":
                new_state["client_selected"] = True
                # 添加明确的成功消息
                new_state["history"].append({
                    "role": "system", 
                    "content": f"客户端已成功设置为 {action_input.get('data_center', 'unknown')}，现在可以查询数据了。"
                })

            # 设置客户端选择状态标记
            if tool_name == "SelectClient" and result.get("status") == "success":
                new_state["client_selected"] = True
                # 添加明确的成功消息
                new_state["history"].append({
                    "role": "system", 
                    "content": f"客户端已成功设置为 {action_input.get('data_center', 'unknown')}，现在可以查询数据了。"
                })
                
            # 在函数内部做一个小调整
            if tool_name == "GetEvents" and result.get("status") == "success":
                new_state["events_fetched"] = True
                new_state["events_data"] = result
                
                # 添加明确的成功消息
                events = result.get("events", [])
                new_state["history"].append({
                    "role": "system", 
                    "content": f"已成功获取 {len(events)} 个地震事件，无需重复查询。请根据用户需求决定是提供总结、生成图表还是下载数据。"
                })

            # GetWaveforms 处理
            if tool_name == "GetWaveforms" and result.get("status") == "success":
                new_state["waveforms_fetched"] = True
                new_state["waveforms_data"] = result
                
                # 添加明确的成功消息
                new_state["history"].append({
                    "role": "system", 
                    "content": "波形数据已成功获取，无需重复查询。请根据用户需求决定是提供总结、绘制图表还是下载数据。"
                })
                
            # 波形数据绘图处理
            if tool_name == "PlotWaveforms" and result.get("status") == "success":
                new_state["plot_path"] = result.get("plot_path")
                new_state["history"].append({
                    "role": "system", 
                    "content": f"波形图已成功生成: {result.get('plot_path')}"
                })
                
            # 波形数据下载处理
            if tool_name == "DownloadWaveforms" and result.get("status") == "success":
                new_state["data_file"] = result.get("data_file")
                new_state["history"].append({
                    "role": "system", 
                    "content": f"波形数据已成功下载到: {result.get('data_file')}"
                })

            # 台站数据处理
            if tool_name == "GetStations" and result.get("status") == "success":
                new_state["stations_fetched"] = True
                new_state["stations_data"] = result
                
                # 添加明确的成功消息
                stations = result.get("stations", [])
                new_state["history"].append({
                    "role": "system", 
                    "content": f"已成功获取 {len(stations)} 个台站信息，无需重复查询。请根据用户需求决定是提供总结、绘制分布图还是下载数据。"
                })
            
            # 台站分布图处理
            if tool_name == "PlotStations" and result.get("status") == "success":
                new_state["plot_path"] = result.get("plot_path")
                new_state["history"].append({
                    "role": "system", 
                    "content": f"台站分布图已成功生成: {result.get('plot_path')}"
                })
            
            # 台站数据下载处理
            if tool_name == "DownloadStations" and result.get("status") == "success":
                new_state["data_file"] = result.get("data_file")
                new_state["history"].append({
                    "role": "system", 
                    "content": f"台站数据已成功下载到: {result.get('data_file')}"
                })    

            # 保存数据文件路径
            if "data_file" in result:
                new_state["data_file"] = result["data_file"]
            
            # 保存图表路径
            if "plot_path" in result:
                new_state["plot_path"] = result["plot_path"]
            
            return new_state
        except Exception as e:
            logger.error(f"{tool_name} 调用失败: {e}")
            new_state["error"] = f"{tool_name} 调用失败: {e}"
            # 添加错误信息到历史
            new_state["history"].append({
                "role": "system", 
                "content": f"错误: {tool_name} 调用失败: {e}"
            })
            return new_state
    
    return node_func

def output_node(state: AgentState) -> Dict:
    """输出节点：格式化最终输出，确保包含必要信息"""
    new_state = state.copy()
    
    # 检查是否有错误
    if state.get("error"):
        new_state["output"] = f"发生错误: {state['error']}"
        return new_state
    
    # 处理最终答案
    if state["action"] == "Final Answer":
        # 判断 action_input 是字符串还是对象
        if isinstance(state["action_input"], dict):
            # 如果是字典，尝试获取消息内容或使用整个字典
            if "message" in state["action_input"]:
                output_text = state["action_input"]["message"]
            elif "text" in state["action_input"]:
                output_text = state["action_input"]["text"]
            else:
                output_text = json.dumps(state["action_input"], ensure_ascii=False)
        else:
            # 如果是字符串，直接使用
            output_text = state["action_input"]
        
        # 补充重要信息
        if state.get("data_file") and "数据文件" not in output_text:
            output_text += f"\n\n数据文件: {state['data_file']}"
        
        if state.get("plot_path") and "图表路径" not in output_text:
            output_text += f"\n图表路径: {state['plot_path']}"
        
        new_state["output"] = output_text
        # 关键修改：确保状态中包含必要的字段
        return {
            **new_state,
            "output": output_text,  # 显式返回output
            "final_answer": True    # 添加标记表示这是最终答案
        }
    
    # 如果是工具结果，格式化输出
    elif state.get("tool_results"):
        result = state["tool_results"]
        output_text = ""
        
        # 添加状态信息
        if "status" in result:
            output_text += f"状态: {result['status']}\n"
        
        # 添加数据文件信息
        data_file = state.get("data_file") or result.get("data_file")
        if data_file:
            output_text += f"数据文件: {data_file}\n"
            
            # 添加数据格式信息（用于下载工具）
            if "format" in result:
                output_text += f"数据格式: {result['format']}\n"
        
        # 添加图表路径
        plot_path = state.get("plot_path") or result.get("plot_path")
        if plot_path:
            output_text += f"图表路径: {plot_path}\n"
        
        # 添加结果消息
        if "message" in result:
            output_text += f"消息: {result['message']}\n"

        # 统一处理工具特定数据
        # 1. 处理地震事件数据
        if state.get("events_data"):
            events = state.get("events_data", {}).get("events", [])
            if events:
                output_text += f"\n找到 {len(events)} 个地震事件"
                # 如果是查询工具，显示详细信息
                if state.get("action") == "GetEvents":
                    output_text += "，详细信息如下：\n"
                    for i, event in enumerate(events[:10]):  # 限制显示条目数
                        output_text += f"\n{i+1}. 时间: {event.get('time')}\n"
                        output_text += f"   震级: {event.get('magnitude')} ({event.get('type', 'unknown')})\n"
                        if "latitude" in event and "longitude" in event:
                            output_text += f"   位置: 纬度 {event.get('latitude'):.4f}, 经度 {event.get('longitude'):.4f}\n"
                        if "depth" in event:
                            output_text += f"   深度: {event.get('depth', 0)/1000:.2f} 公里\n"
                    # 如果事件太多，只显示部分
                    if len(events) > 10:
                        output_text += f"\n... 共 {len(events)} 个事件，仅显示前10个 ...\n"
                else:
                    output_text += "\n"  # 仅显示数量
        
        # 2. 处理波形数据
        if state.get("waveforms_data"):
            traces = state.get("waveforms_data", {}).get("traces", [])
            if traces:
                output_text += f"\n获取了 {len(traces)} 条波形记录"
                # 如果是查询工具，显示详细信息
                if state.get("action") == "GetWaveforms":
                    output_text += "，详细信息如下：\n"
                    for i, trace in enumerate(traces):
                        output_text += f"\n{i+1}. 台站: {trace.get('network')}.{trace.get('station')}.{trace.get('location')}.{trace.get('channel')}\n"
                        output_text += f"   时间范围: {trace.get('starttime')} 至 {trace.get('endtime')}\n"
                        output_text += f"   采样率: {trace.get('sampling_rate')} Hz\n"
                        output_text += f"   数据点数: {trace.get('npts')}\n"
                        output_text += f"   最大幅度: {trace.get('max_amplitude')}\n"
                else:
                    output_text += "\n"  # 仅显示数量

        # 3. 处理台站数据
        if state.get("stations_data"):
            stations = state.get("stations_data", {}).get("stations", [])
            if stations:
                output_text += f"\n获取了 {len(stations)} 个台站数据"
                # 如果是查询工具，显示详细信息
                if state.get("action") == "GetStations":
                    output_text += "，详细信息如下：\n"
                    for i, station in enumerate(stations[:5]):  # 限制显示条目数
                        output_text += f"\n{i+1}. 台站: {station.get('network')}.{station.get('station')}\n"
                        output_text += f"   位置: 纬度 {station.get('latitude'):.4f}, 经度 {station.get('longitude'):.4f}\n"
                        output_text += f"   海拔: {station.get('elevation'):.1f} 米\n"
                        if "site_name" in station and station["site_name"]:
                            output_text += f"   站点名称: {station.get('site_name')}\n"
                        output_text += f"   通道数量: {station.get('channels_count', 0)}\n"
                    # 如果台站太多，只显示部分
                    if len(stations) > 5:
                        output_text += f"\n... 共 {len(stations)} 个台站，仅显示前5个 ...\n"
                else:
                    output_text += "\n"  # 仅显示数量
        
        new_state["output"] = output_text
    
    # 如果是其他情况
    else:
        new_state["output"] = "无可用输出"
    
    return new_state