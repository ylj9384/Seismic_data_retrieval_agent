import os
try:
    import dashscope
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("DASHSCOPE_API_KEY")
    if key: dashscope.api_key = key
except ImportError:
    dashscope = None

CHAT_MODEL = os.getenv("QWEN_CHAT_MODEL","qwen-plus")

def generate(prompt: str, temperature: float=0.2) -> str:
    if dashscope is None or not dashscope.api_key:
        raise RuntimeError("缺少 dashscope 或 API Key")
    rsp = dashscope.Generation.call(
        model=CHAT_MODEL,
        prompt=prompt,
        parameters={"result_format":"text","temperature":temperature}
    )
    if rsp.status_code != 200:
        raise RuntimeError(rsp.message)
    return rsp.output["text"]