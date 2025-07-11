from obspy.clients.fdsn import Client as FDSNClient
from obspy.clients.fdsn import RoutingClient
from obspy import UTCDateTime
import logging
from pydantic import BaseModel, Field
from obspy.clients.fdsn.header import FDSNNoDataException, FDSNException
from typing import List, Dict

logger = logging.getLogger(__name__)

class HybridClient:
    def __init__(self):
        self.available_clients: Dict[str, List[str]] = {
            "routing": ["iris-federator", "eida-routing"],
            "fdsn": ["IRIS", "USGS", "ORFEUS", "GFZ", "INGV"]
        }
        self.current_type: str = "routing"
        self.current_center: str = "iris-federator"
        self._init_client()

    def _init_client(self):
        if self.current_type == "routing":
            self.client = RoutingClient(self.current_center)
        else:
            self.client = FDSNClient(self.current_center)

    def set_client(self, client_type: str, data_center: str) -> str:
        if client_type not in self.available_clients:
            return f"无效的客户端类型: {client_type}"
        if data_center not in self.available_clients[client_type]:
            return f"无效的数据中心: {data_center}"
        self.current_type = client_type
        self.current_center = data_center
        self._init_client()
        return f"客户端已设置为: 类型={client_type}, 数据中心={data_center}"

    def get_current_client(self) -> Dict[str, object]:
        return {
            "client_type": self.current_type,
            "data_center": self.current_center,
            "available_options": self.available_clients
        }

    def robust_call(self, func_name: str, **params):
        """
        动态获取方法并调用，先使用当前客户端调用，失败则依次切换中心或类型重试
        """
        first_err = None
        # 尝试调用方法
        try:
            func = getattr(self.client, func_name)
            return func(**params)
        except Exception as err:
            first_err = err
            logger.warning(f"初次调用 {func_name} 失败: {err}")
        # 同类型其他中心重试
        for center in self.available_clients[self.current_type]:
            if center == self.current_center:
                continue
            self.set_client(self.current_type, center)
            try:
                func = getattr(self.client, func_name)
                logger.info(f"切换到 {self.current_type}/{center} 重试 {func_name}")
                return func(**params)
            except Exception as e:
                logger.warning(f"切换到 {center} 后 {func_name} 失败: {e}")
        # 跨类型重试
        for ctype, centers in self.available_clients.items():
            if ctype == self.current_type:
                continue
            for center in centers:
                self.set_client(ctype, center)
                try:
                    func = getattr(self.client, func_name)
                    logger.info(f"切换至 {ctype}/{center} 重试 {func_name}")
                    return func(**params)
                except Exception as e:
                    logger.warning(f"切换至 {ctype}/{center} 后 {func_name} 失败: {e}")
        # 全部失败，返回最初错误
        return f"所有尝试均失败：{first_err}"

# 初始化 HybridClient 实例
client = HybridClient()

# 参数模型定义
class WaveformParams(BaseModel):
    network: str = Field(description="网络代码")
    station: str = Field(description="台站代码")
    location: str = Field(description="位置代码", default="", )
    channel: str = Field(description="通道代码")
    starttime: str = Field(description="开始时间，ISO8601")
    endtime: str = Field(description="结束时间，ISO8601")

class EventParams(BaseModel):
    starttime: str = Field(description="事件开始时间，ISO8601")
    endtime: str = Field(description="事件结束时间，ISO8601")
    minmagnitude: float = Field(description="最小震级")

class StationParams(BaseModel):
    network: str = Field(description="网络代码")
    station: str = Field(description="台站代码")
    starttime: str = Field(description="开始时间，ISO8601")
    endtime: str = Field(description="结束时间，ISO8601")

class SetClientParams(BaseModel):
    client_type: str = Field(description="客户端类型: routing或fdsn")
    data_center: str = Field(description="数据中心名称")

# 三大工具函数，使用 robust_call 并传入方法名

def retrieve_waveforms(*, network: str, station: str, location: str, channel: str, starttime: str, endtime: str) -> str:
    logger.info("调用 retrieve_waveforms with retry")
    return client.robust_call(
        "get_waveforms",
        network=network,
        station=station,
        location=location,
        channel=channel,
        starttime=UTCDateTime(starttime),
        endtime=UTCDateTime(endtime)
    )


def retrieve_events(*, starttime: str, endtime: str, minmagnitude: float) -> str:
    logger.info("调用 retrieve_events with retry")
    return client.robust_call(
        "get_events",
        starttime=UTCDateTime(starttime),
        endtime=UTCDateTime(endtime),
        minmagnitude=minmagnitude
    )


def retrieve_stations(*, network: str, station: str, starttime: str, endtime: str) -> str:
    logger.info("调用 retrieve_stations with retry")
    return client.robust_call(
        "get_stations",
        network=network,
        station=station,
        starttime=UTCDateTime(starttime),
        endtime=UTCDateTime(endtime),
        level="response"
    )

# Client 管理工具

def set_client(*, client_type: str, data_center: str) -> str:
    """设置客户端配置"""
    logger.info(f"调用 set_client: {client_type}/{data_center}")
    return client.set_client(client_type, data_center)


def get_client_info() -> str:
    """获取当前客户端信息"""
    info = client.get_current_client()
    return (
        f"当前客户端: {info['client_type']}({info['data_center']})\n"
        f"可用服务: routing[{', '.join(info['available_options']['routing'])}] | "
        f"fdsn[{', '.join(info['available_options']['fdsn'])}]"
    )