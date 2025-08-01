import logging
import json
from data_retrieval.agent_initializer import build_agent
from data_retrieval.state import AgentState
import pprint
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 构建LangGraph代理
    agent = build_agent()
    print("欢迎使用地震数据智能检索 Agent！请输入您的问题：")
    
    # 初始化状态
    state = AgentState(
        {
            "user_input": "",
            "history": [],
            "action": None,
            "action_input": None,
            "tool_results": None,
            "output": None,
            "data_file": None,
            "plot_path": None,
            "events_data": None,
            "client_selected": False,
            "events_fetched": False,
            "error": None
        }
    )
    
    while True:
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
        # 注意：不重置 data_file, plot_path 等持久状态，以便后续对话可引用
        
        # 调用agent
        logger.info(f"用户输入: {user_input}")
        try:
            result = agent.invoke(state)
    
            # 详细打印结果（调试用）
            print("\nDEBUG - Agent 返回结果:")
            pprint.pprint(result)
            
           # 直接从结果中提取输出
            if "output" in result:
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
            
            # 更新状态
            for key, value in result.items():
                state[key] = value
        except Exception as e:
            logger.error(f"Agent执行出错: {e}")
            print(f"\n执行出错: {e}")
            logger.error(f"详细错误信息: {traceback.format_exc()}")

if __name__ == "__main__":
    main()