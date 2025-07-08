from langchain_core.messages import SystemMessage, HumanMessage
from agent.llm import get_kimi_llm

# 构造测试消息
messages = [
    SystemMessage(content="请用如下格式输出：\n{\"action\": \"TestAction\", \"action_input\": {\"key\": \"value\"}}"),
    HumanMessage(content="请输出 action 和 action_input。")
]

# 获取模型并发送消息
llm = get_kimi_llm()
response = llm.invoke(messages)

# 打印模型响应
print("模型输出：")
print(response.content)
