from langchain.tools import StructuredTool
from .tools import (
    retrieve_waveforms, retrieve_events, retrieve_stations,
    set_client, get_client_info,
    WaveformParams, EventParams, StationParams, SetClientParams
)

def get_tools():
    """返回符合LangChain标准的工具集合，包括客户端选择及地震工具"""
    return [
        StructuredTool.from_function(
            name="SelectClient",
            func=set_client,
            description="设置客户端类型和数据中心",
            args_schema=SetClientParams
        ),
        StructuredTool.from_function(
            name="GetClientInfo",
            func=get_client_info,
            description="获取当前客户端配置信息",
            # 无需参数时可不指定 schema
            args_schema=None
        ),
        StructuredTool.from_function(
            name="GetWaveforms",
            func=retrieve_waveforms,
            description="获取波形数据（含自动重试机制）",
            args_schema=WaveformParams
        ),
        StructuredTool.from_function(
            name="GetEvents",
            func=retrieve_events,
            description="获取地震事件（含自动重试机制）",
            args_schema=EventParams
        ),
        StructuredTool.from_function(
            name="GetStations",
            func=retrieve_stations,
            description="获取台站信息（含自动重试机制）",
            args_schema=StationParams
        )
    ]