from obspy.clients.fdsn import Client as FDSNClient
from obspy import UTCDateTime
import logging
from pydantic import BaseModel, Field
from obspy.clients.fdsn.header import FDSNNoDataException, FDSNException

logger = logging.getLogger(__name__)
client = FDSNClient("IRIS")

# 参数模型保持不变
class WaveformParams(BaseModel):
    network: str = Field(description="网络代码")
    station: str = Field(description="台站代码")
    location: str = Field(description="位置代码", default="")
    channel: str = Field(description="通道代码")
    starttime: str = Field(description="开始时间，格式为ISO8601字符串")
    endtime: str = Field(description="结束时间，格式为ISO8601字符串")

class EventParams(BaseModel):
    starttime: str = Field(description="事件开始时间，格式为ISO8601字符串")
    endtime: str = Field(description="事件结束时间，格式为ISO8601字符串")
    minmagnitude: float = Field(description="最小震级")

class StationParams(BaseModel):
    network: str = Field(description="网络代码")
    station: str = Field(description="台站代码")
    starttime: str = Field(description="开始时间，格式为ISO8601字符串")
    endtime: str = Field(description="结束时间，格式为ISO8601字符串")

def retrieve_waveforms(
    *,
    network: str,
    station: str,
    location: str,
    channel: str,
    starttime: str,
    endtime: str
) -> str:
    """获取波形数据"""
    logger.info(f"调用 retrieve_waveforms: network={network}, station={station}, location={location}, channel={channel}, start={starttime}, end={endtime}")
    
    try:
        # 尝试获取波形数据
        st = client.get_waveforms(
            network=network,
            station=station,
            location=location,
            channel=channel,
            starttime=UTCDateTime(starttime),
            endtime=UTCDateTime(endtime),
        )
        return f"成功获取波形数据，包含 {len(st)} 个 trace，持续时间为 {st[0].stats.starttime} 到 {st[0].stats.endtime}"
    
    except FDSNNoDataException:
        # 处理无数据情况
        error_msg = f"未找到符合条件的数据: {network}.{station}.{location}.{channel} ({starttime} 到 {endtime})"
        logger.warning(error_msg)
        
        # 建议可能的解决方案
        suggestions = [
            "检查时间范围是否在台站运行期间",
            "尝试不同的位置代码（如 '00' 而不是空字符串）",
            "尝试不同的通道（如 BH1, BH2 而不是 BHZ）",
            "确认台站在该时间段是否运行"
        ]
        return error_msg + "\n建议: " + "; ".join(suggestions)
    
    except FDSNException as e:
        # 处理FDSN服务错误
        error_msg = f"FDSN服务错误: {str(e)}"
        logger.error(error_msg)
        suggestions = [
            "检查网络连接",
            "尝试稍后重试",
            "确认服务器状态"
        ]
        return error_msg + "\n建议: " + "; ".join(suggestions)
    
    except Exception as e:
        # 处理其他异常
        error_msg = f"获取波形数据时出错: {str(e)}"
        logger.exception(error_msg)
        return error_msg

def retrieve_events(
    *,
    starttime: str,
    endtime: str,
    minmagnitude: float
) -> str:
    """获取地震事件"""
    logger.info(f"调用 retrieve_events: start={starttime}, end={endtime}, minmag={minmagnitude}")
    
    try:
        # 尝试获取地震事件
        cat = client.get_events(
            starttime=UTCDateTime(starttime),
            endtime=UTCDateTime(endtime),
            minmagnitude=minmagnitude,
        )
        
        if not cat:
            return f"在 {starttime} 到 {endtime} 期间未找到震级大于 {minmagnitude} 的地震事件"
        
        return f"获取到 {len(cat)} 个事件。第一个事件时间：{cat[0].origins[0].time}，震级：{cat[0].magnitudes[0].mag}"
    
    except FDSNNoDataException:
        # 处理无数据情况
        error_msg = f"未找到符合条件的地震事件: {starttime} 到 {endtime}，最小震级 {minmagnitude}"
        logger.warning(error_msg)
        
        # 建议可能的解决方案
        suggestions = [
            "放宽时间范围",
            "降低最小震级要求",
            "尝试不同的数据中心（可能需要更换客户端）"
        ]
        return error_msg + "\n建议: " + "; ".join(suggestions)
    
    except FDSNException as e:
        # 处理FDSN服务错误
        error_msg = f"FDSN服务错误: {str(e)}"
        logger.error(error_msg)
        suggestions = [
            "检查网络连接",
            "尝试稍后重试",
            "确认服务器状态"
        ]
        return error_msg + "\n建议: " + "; ".join(suggestions)
    
    except Exception as e:
        # 处理其他异常
        error_msg = f"获取地震事件时出错: {str(e)}"
        logger.exception(error_msg)
        return error_msg

def retrieve_stations(
    *,
    network: str,
    station: str,
    starttime: str,
    endtime: str
) -> str:
    """获取台站信息"""
    logger.info(f"调用 retrieve_stations: network={network}, station={station}, start={starttime}, end={endtime}")
    
    try:
        # 尝试获取台站信息
        inv = client.get_stations(
            network=network,
            station=station,
            starttime=UTCDateTime(starttime),
            endtime=UTCDateTime(endtime),
            level="response",
        )
        
        if not inv:
            return f"未找到符合条件的台站信息: {network}.{station} ({starttime} 到 {endtime})"
        
        return f"获取到 {len(inv)} 个台网，包含台站 {inv[0][0].code}"
    
    except FDSNNoDataException:
        # 处理无数据情况
        error_msg = f"未找到符合条件的台站信息: {network}.{station} ({starttime} 到 {endtime})"
        logger.warning(error_msg)
        
        # 建议可能的解决方案
        suggestions = [
            "放宽时间范围",
            "使用通配符（如 'IU.*'）",
            "尝试不同的数据中心（可能需要更换客户端）"
        ]
        return error_msg + "\n建议: " + "; ".join(suggestions)
    
    except FDSNException as e:
        # 处理FDSN服务错误
        error_msg = f"FDSN服务错误: {str(e)}"
        logger.error(error_msg)
        suggestions = [
            "检查网络连接",
            "尝试稍后重试",
            "确认服务器状态"
        ]
        return error_msg + "\n建议: " + "; ".join(suggestions)
    
    except Exception as e:
        # 处理其他异常
        error_msg = f"获取台站信息时出错: {str(e)}"
        logger.exception(error_msg)
        return error_msg