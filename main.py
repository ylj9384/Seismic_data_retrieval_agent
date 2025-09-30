import logging
import os
import json
import pprint
import traceback
from langchain_openai import ChatOpenAI
from orchestrator.system import OrchestratorSystem
from config.llm import get_kimi_llm, get_qwen_llm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # 初始化LLM
    # llm = get_kimi_llm()
    llm = get_qwen_llm()
    
    # 创建编排系统
    system = OrchestratorSystem(llm=llm, mode="supervisor")
    
    print("欢迎使用地震学多智能体系统！请输入您的问题：")
    print("输入'exit'或'quit'退出系统。")
    
    # 初始化上下文
    context = {}
    
    while True:
        try:
            user_input = input(">>> ")
            if user_input.lower() in ["exit", "quit"]:
                print("已退出。")
                break
            
            logger.info(f"用户输入: {user_input}")
            
            # 调用编排系统处理查询
            result = system.process(user_input, context)
            
            # 详细打印结果（调试用）
            print("\nDEBUG - 编排器返回结果:")
            pprint.pprint(result)
            
            # 显示结果
            if "output" in result:
                print("\n结果:\n", result["output"])
            else:
                print("\n结果: 无输出")
            
            # 显示文件或图表(如果有)
            for key, value in result.items():
                if key.endswith(("_file", "_path")) and value:
                    print(f"\n{key}: {value}")
                    # 将文件路径添加到用户下一个输入
                    if "last_data_file" in result and "波形" in user_input.lower():
                        context["waveform_file"] = result["last_data_file"]
                        # print(f"\n[系统] 已记录波形文件路径: {result['last_data_file']}")
            
            # 如有错误，显示错误信息
            if "error" in result:
                print(f"\n错误: {result['error']}")
                
        except Exception as e:
            logger.error(f"系统执行出错: {e}")
            print(f"\n执行出错: {e}")
            logger.error(f"详细错误信息: {traceback.format_exc()}")


if __name__ == "__main__":
    main()