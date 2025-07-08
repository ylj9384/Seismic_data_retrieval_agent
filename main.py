from agent.agent_initializer import build_agent
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    agent = build_agent()
    print("欢迎使用地震数据智能检索 Agent！请输入您的问题：")
    while True:
        user_input = input(">>> ")
        if user_input.lower() in ["exit", "quit"]:
            print("已退出。"); break
        logger.info(f"用户输入: {user_input}")
        response = agent.invoke({"input": user_input})  
        print("结果：\n", response["output"])

if __name__ == "__main__":
    main()