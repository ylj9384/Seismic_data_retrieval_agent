from typing import Dict, Callable, Any
from .tools import (
    retrieve_waveforms, retrieve_events, retrieve_stations,
    set_client, get_client_info, plot_catalog, download_catalog_data,
    download_waveforms, plot_waveforms,
    download_stations, plot_stations,  explain_location_codes, # 添加新工具
    EventParams, SetClientParams, CatalogParam,
    DownloadCatalogParams, WaveformDataParam, DownloadWaveformsParams, PlotWaveformsParams,
    StationDataParam, DownloadStationsParams, PlotStationsParams  # 添加新参数模型
)

def get_tools() -> Dict[str, Callable]:
    """
    返回工具函数字典，供 LangGraph 节点引用
    """
    return {
        "SelectClient": set_client,
        "GetClientInfo": get_client_info,
        "GetWaveforms": retrieve_waveforms,
        "GetEvents": retrieve_events,
        "GetStations": retrieve_stations,
        "PlotCatalog": plot_catalog,
        "DownloadCatalog": download_catalog_data,
        "DownloadWaveforms": download_waveforms,
        "PlotWaveforms": plot_waveforms,
        "DownloadStations": download_stations,  # 新增
        "PlotStations": plot_stations,  # 新增
        "ExplainLocationCodes": explain_location_codes,
    }

def get_tool_descriptions() -> Dict[str, str]:
    """
    返回工具描述字典，供 LLMNode 提示词使用
    """
    return {
        "SelectClient": "设置客户端类型和数据中心，参数：client_type, data_center",
        "GetClientInfo": "获取当前客户端配置信息，无需参数",
        "GetWaveforms": "获取波形数据信息，参数：network, station, location, channel, starttime, endtime",
        "GetEvents": "获取地震事件，参数：starttime, endtime, minmagnitude",
        "GetStations": "获取台站信息，参数：network, station, starttime, endtime",
        "PlotCatalog": "生成地震事件分布图表，参数：catalog_data",
        "DownloadCatalog": "下载地震目录数据，参数：catalog_data, format",
        "DownloadWaveforms": "下载波形数据，参数：waveform_data, format",
        "PlotWaveforms": "绘制波形数据图表，参数：waveform_data, filter_type, freqmin, freqmax",
        "DownloadStations": "下载台站数据，参数：station_data, format",  # 新增
        "PlotStations": "绘制台站分布图，参数：station_data, map_type",  # 新增
    }

def get_tool_param_models() -> Dict[str, Any]:
    """
    返回工具参数模型，供参数校验使用
    """
    return {
        "SelectClient": SetClientParams,
        "GetWaveforms": WaveformDataParam,
        "GetEvents": EventParams,
        "GetStations": StationDataParam,
        "PlotCatalog": CatalogParam,
        "DownloadCatalog": DownloadCatalogParams,
        "DownloadWaveforms": DownloadWaveformsParams,
        "PlotWaveforms": PlotWaveformsParams,
        "DownloadStations": DownloadStationsParams,
        "PlotStations": PlotStationsParams,
    }