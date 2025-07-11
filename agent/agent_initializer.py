from langchain.agents import initialize_agent, AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate 
from langchain_core.messages import SystemMessage, HumanMessage 
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from .llm import get_kimi_llm
from .tool_registry import get_tools

# 定义输出格式规范
response_schemas = [
    ResponseSchema(name="action", description="要调用的工具名称"),
    ResponseSchema(name="action_input", description="工具参数，必须是JSON格式")
]
# 基于response_schemas创建输出解析器，负责将LLM文本输出转为结构化数据
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
# 自动生成的格式化指南，人类可读的输出规范说明
format_instructions = output_parser.get_format_instructions()

# 强化提示词 - 添加明确的JSON格式约束和错误示例
system = SystemMessage(content=f"""
    你是一个地震数据助手，具有以下工具可供调用：

    1. GetWaveforms: 获取波形数据，参数包括 network, station, location, channel, starttime, endtime。
    2. GetEvents: 获取地震事件，参数包括 starttime, endtime, minmagnitude。
    3. GetStations: 获取台站信息，参数包括 network, station, starttime, endtime。
    4. SelectClient: 选择客户端类型和数据中心，参数包括 client_type, data_center
    
    请根据用户问题判断是否需要调用工具：
    - 如果用户询问地震事件、台站信息、或波形数据，请调用合适的工具；
    - 如果用户只是打招呼、寒暄或问天气等日常交流，请直接用自然语言回复，不要调用任何工具；
    - 工具名称只能为以上列出的内容，不要凭空创造新的工具；

    关键指令：
    1. 调用工具时，必须输出完整的JSON对象
    2. action_input 必须且只能是JSON对象
    3. 绝对不要使用任何形式的字符串参数（包括逗号分隔或URL编码）
    4. 输出必须能被Python的json.loads()函数直接解析
    5. 当遇到数据获取失败时，可尝试切换客户端
    6. 需要特定数据中心时调用SelectClient工具

    ！！！参数名称必须精确匹配！！！
    - GetEvents 参数: starttime, endtime, minmagnitude
    - GetWaveforms 参数: network, station, location, channel, starttime, endtime
    - GetStations 参数: network, station, starttime, endtime
    - SelectClient 参数: client_type, data_center

    有效输出示例：
    {{
        "action": "SelectClient",
        "action_input": {{
            "client_type": "fdsn",
            "data_center": "USGS"
        }}
    }}
                       
    {{
        "action": "GetEvents",
        "action_input": {{ 
            "starttime": "2020-01-01",
            "endtime": "2020-01-02",
            "minmagnitude": 5.0 
        }}
    }}

    {{
        "action": "GetWaveforms",
        "action_input": {{ 
            "network": "IU",
            "station": "ANMO",
            "location": "",
            "channel": "BHZ",
            "starttime": "2020-01-01T00:00:00",
            "endtime": "2020-01-02T00:00:00"
        }}
    }}

    错误示例1（字符串参数）:
    {{"action": "GetEvents", "action_input": "2020-01-01,2020-01-02,5.0"}}

    错误示例2（键名错误）:
    {{"action": "GetEvents", "action_input": {{
        "start_time": "2020-01-01",  # 错误! 应该是 'starttime'
        "end_time": "2020-01-02",    # 错误! 应该是 'endtime'
        "magnitude": 5.0             # 错误! 应该是 'minmagnitude'
    }}}}

    格式说明:
    {format_instructions}

    严格要求：
    1. `action_input` 必须是 JSON 对象，不能是字符串。
    2. 返回内容必须可以直接被 `json.loads()` 解析。
    3. 不要输出多余内容，如「我将调用...」。
    4. 参数名称必须精确匹配上述规范。
    5. 调用客户端失败请使用另一种客户端或者数据中心
    6. 在第一次任何查询前，请先调用 SelectClient 设置 client_type 与 data_center

    请始终只返回符合要求的 JSON。
    """)


def build_agent():
    # 结构化对话提示模板，定义与LLM交互的消息序列
    prompt = ChatPromptTemplate.from_messages([system, HumanMessage(content="{input}")])
    # 记录用户与Agent的完整对话历史；支持多轮地震数据分析
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # 使用支持结构化输出的代理类型
    agent = initialize_agent(
        tools=get_tools(),
        llm=get_kimi_llm(),
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        prompt=prompt,
        memory=memory,
        output_parser=output_parser,
        # 用于处理LLM输出解析失败的情况
        handle_parsing_errors=True,
        max_iterations=3  # 限制迭代次数避免无限循环
    )
    
    return agent