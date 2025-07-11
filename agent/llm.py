import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def get_kimi_llm():
    return ChatOpenAI(
	base_url="https://api.moonshot.cn/v1",
	model="moonshot-v1-8k",
	openai_api_key=os.environ.get("OPENAI_API_KEY")
)
