import logging
import json
from phase_detection.agent_initializer import build_agent
from phase_detection.state import PhaseDetectionState
import pprint
import traceback
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # 构建震相检测 LangGraph 代理
    agent = build_agent()
    print("欢迎使用地震学震相拾取与事件检测 Agent！请输入您的问题：")
    print("输入'exit'或'quit'退出系统。")
    print("示例命令:")
    print("1. 列出可用的震相检测模型")
    print("2. 使用PhaseNet模型检测波形文件中的震相并绘制图像")
    print("3. 评估震相检测质量")
    print("4. 比较不同模型的震相拾取结果")
    
    # 初始化状态
    state = PhaseDetectionState(
        {
            "user_input": "",
            "history": [],
            "action": None,
            "action_input": None,
            "tool_results": None,
            "detection_id": None,
            "detection_results": None,
            "plot_path": None,
            "output": None,
            "error": None
        }
    )
    
    while True:
        try:
            user_input = input(">>> ")
            if user_input.lower() in ["exit", "quit"]:
                print("已退出。")
                break
            
            # 更新状态中的用户输入
            state["user_input"] = user_input
            
            # 添加到历史
            state["history"].append({"role": "user", "content": user_input})
            
            # 重置单次交互的状态
            state["action"] = None
            state["action_input"] = None
            state["tool_results"] = None
            state["error"] = None
            # 注意：不重置 detection_id, plot_path 等持久状态，以便后续对话可引用
            
            # 调用agent
            logger.info(f"用户输入: {user_input}")
            result = agent.invoke(state)
            
            # 详细打印结果（调试用）
            print("\nDEBUG - Agent 返回结果:")
            # 只打印部分关键字段，避免输出过多
            debug_result = {
                "action": result.get("action"),
                "detection_id": result.get("detection_id"),
                "plot_path": result.get("plot_path"),
                "error": result.get("error")
            }
            if "tool_results" in result and result["tool_results"]:
                debug_result["tool_status"] = result["tool_results"].get("status")
                if "message" in result["tool_results"]:
                    debug_result["tool_message"] = result["tool_results"]["message"]
                
            pprint.pprint(debug_result)
            
            # 显示结果
            if "output" in result and result["output"]:
                print("\n结果:\n", result["output"])
            else:
                # 尝试从action_input中提取
                if result.get("action") == "Final Answer":
                    action_input = result.get("action_input")
                    if isinstance(action_input, str):
                        print("\n结果:\n", action_input)
                    elif isinstance(action_input, dict):
                        print("\n结果:\n", json.dumps(action_input, ensure_ascii=False))
                    else:
                        print("\n结果: 无法解析输出")
                else:
                    print("\n结果: 无输出")
            
            # # 显示图表路径
            # if result.get("plot_path") and os.path.exists(result.get("plot_path")):
            #     print(f"\n图表已生成: {result.get('plot_path')}")
            
            # 更新状态
            for key, value in result.items():
                state[key] = value
                
        except Exception as e:
            logger.error(f"Agent执行出错: {e}")
            print(f"\n执行出错: {e}")
            logger.error(f"详细错误信息: {traceback.format_exc()}")

if __name__ == "__main__":
    main()