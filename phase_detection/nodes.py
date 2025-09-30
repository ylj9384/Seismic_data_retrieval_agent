import logging
import json
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from .state import PhaseDetectionState
from config.llm import get_kimi_llm, get_qwen_llm  # 复用现有的LLM
from .prompt_templates import create_system_prompt

logger = logging.getLogger(__name__)

def llm_node(state: PhaseDetectionState) -> Dict:
    """LLM 节点：处理用户输入，决定下一步行动"""
    # llm = get_kimi_llm()
    llm = get_qwen_llm()
    
    # 构建历史消息
    messages = [SystemMessage(content=create_system_prompt())]

    # 添加历史消息
    for msg in state["history"]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
        elif msg["role"] == "system":
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

        # 创建新状态
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
    def node_func(state: PhaseDetectionState) -> Dict:
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
            
            # 添加工具执行结果到历史记录
            # 限制JSON大小以防止历史记录过大
            result_copy = result.copy() if isinstance(result, dict) else {}
            if isinstance(result_copy, dict):
                # 移除可能很大的字段
                for key in ['data_cache', 'picks', 'detections']:
                    if key in result_copy:
                        result_copy[key] = f"[已省略 {len(result_copy[key]) if isinstance(result_copy[key], list) else '数据'} 以减小大小]"
            
            result_message = f"工具 {tool_name} 执行结果: {json.dumps(result_copy, ensure_ascii=False)}"
            new_state["history"].append({"role": "system", "content": result_message})
            
            # 统一处理所有工具的结果
            if result.get("status") == "success":
                # 保存重要的元数据
                if "detection_id" in result:
                    new_state["detection_id"] = result.get("detection_id")
                
                if "data_cache" in result:
                    new_state["data_cache"] = result["data_cache"]
                
                if "plot_path" in result:
                    new_state["plot_path"] = result["plot_path"]
                    plot_message = f"绘图已完成: {result['plot_path']}"
                    new_state["history"].append({"role": "system", "content": plot_message})
                
                # 为DetectAndPlotPhases工具添加更详细的成功消息
                if tool_name == "DetectAndPlotPhases":
                    success_message = (
                        f"震相拾取与绘图完成，检测到{result.get('picks_count', 0)}个震相和"
                        f"{result.get('detections_count', 0)}个事件。"
                    )
                    if "plot_path" in result:
                        success_message += f"\n图像已保存至: {result['plot_path']}"
                    
                    new_state["history"].append({
                        "role": "system", 
                        "content": success_message
                    })
            
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

def output_node(state: PhaseDetectionState) -> Dict:
    """输出节点：格式化最终输出"""
    new_state = state.copy()
    
    # 检查是否有错误
    if state.get("error"):
        new_state["output"] = f"发生错误: {state['error']}"
        return new_state
    
    # 处理最终答案
    if state["action"] == "Final Answer":
        # 判断 action_input 是字符串还是对象
        if isinstance(state["action_input"], dict):
            if "message" in state["action_input"]:
                output_text = state["action_input"]["message"]
            elif "text" in state["action_input"]:
                output_text = state["action_input"]["text"]
            else:
                output_text = json.dumps(state["action_input"], ensure_ascii=False)
        else:
            output_text = state["action_input"]
        
        # 补充重要信息
        if state.get("plot_path") and "图" not in output_text.lower():
            output_text += f"\n\n图表已保存至: {state['plot_path']}"
        
        new_state["output"] = output_text
        return {
            **new_state,
            "output": output_text,
            "final_answer": True
        }
    
    # 如果是工具结果，格式化输出
    elif state.get("tool_results"):
        result = state["tool_results"]
        output_text = ""
        
        # 添加状态信息
        if "status" in result:
            output_text += f"状态: {result['status']}\n"
        
        # 添加图表路径
        if "plot_path" in result:
            output_text += f"图表已保存至: {result['plot_path']}\n"
        
        # 添加结果消息
        if "message" in result:
            output_text += f"消息: {result['message']}\n"
        
        # 为DetectAndPlotPhases结果提供更好的格式化
        if "picks" in result and "picks_count" in result:
            picks = result["picks"]
            output_text += f"\n找到 {result['picks_count']} 个震相:\n"
            for i, pick in enumerate(picks[:5]):  # 限制显示条目数
                phase = pick.get('phase', '未知')
                time = pick.get('time', '未知')
                prob = pick.get('probability', 'N/A')
                prob_str = f", 概率: {prob:.2f}" if isinstance(prob, (int, float)) else ""
                output_text += f"{i+1}. {phase}波: {time}{prob_str}\n"
            if len(picks) > 5:
                output_text += f"... 共 {len(picks)} 个震相，仅显示前5个 ...\n"
        
        # 添加事件检测特定信息
        if "detections" in result and "detections_count" in result:
            detections = result["detections"]
            output_text += f"\n找到 {result['detections_count']} 个事件:\n"
            for i, det in enumerate(detections[:3]):  # 限制显示条目数
                start_time = det.get("start_time", det.get("time", "未知"))
                end_time = det.get("end_time", "未知")
                output_text += f"{i+1}. 事件: {start_time} 至 {end_time}\n"
            if len(detections) > 3:
                output_text += f"... 共 {len(detections)} 个事件，仅显示前3个 ...\n"
        
        # 添加概率信息
        if "probabilities" in result:
            probs = result["probabilities"]
            if probs:  # 只有在有概率数据时才显示
                output_text += "\n概率信息:\n"
                for key, value in probs.items():
                    formatted_key = key.replace("_max_probability", "最大概率").replace("_", " ")
                    output_text += f"{formatted_key}: {value:.4f}\n"
        
        new_state["output"] = output_text
    
    # 如果是其他情况
    else:
        new_state["output"] = "无可用输出"
    
    return new_state