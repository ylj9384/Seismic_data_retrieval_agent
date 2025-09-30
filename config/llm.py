import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def get_kimi_llm():
    return ChatOpenAI(
	base_url="https://api.moonshot.cn/v1",
	# model="moonshot-v1-8k",
    model="kimi-k2-0711-preview",
	openai_api_key=os.environ.get("OPENAI_API_KEY")
)

def get_qwen_llm():
    return ChatOpenAI(
	base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    # model="text-embedding-v1",
	openai_api_key=os.environ.get("QWEN_API_KEY")
)
