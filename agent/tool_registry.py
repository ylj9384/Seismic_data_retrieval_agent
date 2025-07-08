from langchain.tools import StructuredTool
from .tools import retrieve_waveforms, retrieve_events, retrieve_stations
from .tools import WaveformParams, EventParams, StationParams  # 导入参数模型

def get_tools():
    """返回符合LangChain标准的工具集合"""
    return [
        StructuredTool.from_function(
            name="GetWaveforms",
            func=retrieve_waveforms,
            description="获取波形数据",
            args_schema=WaveformParams
        ),
        StructuredTool.from_function(
            name="GetEvents",
            func=retrieve_events,
            description="获取地震事件",
            args_schema=EventParams
        ),
        StructuredTool.from_function(
            name="GetStations",
            func=retrieve_stations,
            description="获取台站信息",
            args_schema=StationParams
        )
    ]