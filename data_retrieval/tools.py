from obspy.clients.fdsn import Client as FDSNClient
from obspy.clients.fdsn import RoutingClient
from obspy import UTCDateTime
import logging
import tempfile
import os
import sys
import subprocess
from typing import Dict, List, Any
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# 参数模型定义
class WaveformDataParam(BaseModel):
    waveform_data: str = Field(description="波形数据标识符，格式：network|station|location|channel|starttime|endtime")

class DownloadWaveformsParams(BaseModel):
    waveform_data: str = Field(description="波形数据标识符，格式：network|station|location|channel|starttime|endtime")
    format: str = Field(description="数据格式: MSEED, SAC, SEGY, WAV", default="MSEED")

class PlotWaveformsParams(BaseModel):
    waveform_data: str = Field(description="波形数据标识符，格式：network|station|location|channel|starttime|endtime")
    filter_type: str = Field(description="滤波类型: none, bandpass, lowpass, highpass", default="none")
    freqmin: float = Field(description="最低频率，用于bandpass和highpass滤波", default=0.0)
    freqmax: float = Field(description="最高频率，用于bandpass和lowpass滤波", default=0.0)

class EventParams(BaseModel):
    starttime: str = Field(description="事件开始时间，ISO8601")
    endtime: str = Field(description="事件结束时间，ISO8601")
    minmagnitude: float = Field(description="最小震级")

class CatalogParam(BaseModel):
    catalog_data: str = Field(description="地震目录数据标识符，格式：starttime|endtime|minmagnitude")

class DownloadCatalogParams(BaseModel):
    catalog_data: str = Field(description="地震目录数据标识符，格式：starttime|endtime|minmagnitude")
    format: str = Field(description="数据格式: QUAKEML, CSV, JSON", default="QUAKEML")

class SetClientParams(BaseModel):
    client_type: str = Field(description="客户端类型: routing或fdsn")
    data_center: str = Field(description="数据中心名称")

class StationDataParam(BaseModel):
    station_data: str = Field(description="台站数据标识符，格式：network|station|starttime|endtime")

class DownloadStationsParams(BaseModel):
    station_data: str = Field(description="台站数据标识符，格式：network|station|starttime|endtime")
    format: str = Field(description="数据格式: STATIONXML, CSV, JSON", default="STATIONXML")

class PlotStationsParams(BaseModel):
    station_data: str = Field(description="台站数据标识符，格式：network|station|starttime|endtime")
    map_type: str = Field(description="地图类型: global, regional, local", default="global")




# HybridClient类保持不变，这是良好设计
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

    def set_client(self, client_type: str, data_center: str) -> Dict[str, Any]:
        """设置客户端类型和数据中心"""
        if client_type not in self.available_clients:
            return {"status": "error", "message": f"无效的客户端类型: {client_type}"}
        if data_center not in self.available_clients[client_type]:
            return {"status": "error", "message": f"无效的数据中心: {data_center}"}
        
        self.current_type = client_type
        self.current_center = data_center
        self._init_client()
        
        return {
            "status": "success", 
            "client_type": client_type,
            "data_center": data_center,
            "message": f"客户端已设置为: 类型={client_type}, 数据中心={data_center}"
        }

    def get_current_client(self) -> Dict[str, Any]:
        """获取当前客户端信息"""
        return {
            "status": "success",
            "client_type": self.current_type,
            "data_center": self.current_center,
            "available_options": self.available_clients,
            "message": f"当前客户端: {self.current_type}({self.current_center})"
        }

    def robust_call(self, func_name: str, **params):
        """动态获取方法并调用，先使用当前客户端调用，失败则依次切换中心或类型重试"""
        first_err = None
        # 尝试调用方法
        try:
            # 等价于 func = self.client.get_waveforms
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
        return {"status": "error", "message": f"所有尝试均失败：{first_err}"}

# 初始化 HybridClient 实例
client = HybridClient()

# 工具函数定义 - 规范化返回值为字典，便于LangGraph处理
# LangGraph 框架下不需要添加 @tool 装饰器，它采用了更灵活、更明确的节点和工具引用方式。

def retrieve_waveforms(network: str, station: str, location: str, channel: str, starttime: str, endtime: str) -> Dict[str, Any]:
    """获取波形数据信息"""
    logger.info(f"调用 retrieve_waveforms: {network}.{station}.{location}.{channel}")
    
     # 解析位置代码的自然语言描述
    location_mapping = {
        "主传感器": "00",
        "主要位置": "00", 
        "标准位置": "00",
        "备用传感器": "10",
        "次要位置": "10",
        "未指定位置": "",
        "默认位置": "",
        "空位置": ""
    }
    
    # 如果location是自然语言描述，转换为代码
    if location in location_mapping:
        actual_location = location_mapping[location]
        logger.info(f"位置描述'{location}'被解析为位置代码'{actual_location}'")
    else:
        actual_location = location
    
    try:
        result = client.robust_call(
            "get_waveforms",
            network=network,
            station=station,
            location=location,
            channel=channel,
            starttime=UTCDateTime(starttime),
            endtime=UTCDateTime(endtime)
        )
        
        # 格式化波形数据信息
        traces_info = []
        for tr in result:
            stats = tr.stats
            # 关键修改: 确保所有数值类型都转换为标准Python类型
            traces_info.append({
                "network": stats.network,
                "station": stats.station,
                "location": stats.location,
                "channel": stats.channel,
                "starttime": stats.starttime.isoformat(),
                "endtime": stats.endtime.isoformat(),
                "sampling_rate": float(stats.sampling_rate),  # 转换为Python float
                "npts": int(stats.npts),  # 转换为Python int
                "delta": float(stats.delta),  # 转换为Python float
                "max_amplitude": float(max(abs(tr.data))) if len(tr.data) > 0 else 0.0  # 转换为Python float
            })

        # 保存波形数据参数，供下载和绘图使用
        waveform_data = f"{network}|{station}|{location}|{channel}|{starttime}|{endtime}"
        
        return {
            "status": "success",
            "traces_count": len(result),
            "time_range": f"{starttime} 至 {endtime}",
            "waveform_data": waveform_data,
            "traces": traces_info,
            "message": f"成功获取 {network}.{station}.{location}.{channel} 的波形数据信息"
        }
    except Exception as e:
        logger.error(f"获取波形数据失败: {e}")
        return {"status": "error", "message": f"获取波形数据失败: {str(e)}"}

def download_waveforms(waveform_data: str, format: str = "MSEED") -> Dict[str, Any]:
    """下载波形数据并保存为文件
    
    Args:
        waveform_data: 格式为"network|station|location|channel|starttime|endtime"的字符串
        format: 输出格式，默认为MSEED，可选值：MSEED, SAC, SEGY, WAV
        
    Returns:
        包含下载结果信息的字典
    """
    logger.info(f"调用 download_waveforms: {waveform_data}, 格式: {format}")
    try:
        # 解析参数
        network, station, location, channel, starttime, endtime = waveform_data.split("|")
        
        # 获取数据
        st = client.robust_call(
            "get_waveforms",
            network=network,
            station=station,
            location=location,
            channel=channel,
            starttime=UTCDateTime(starttime),
            endtime=UTCDateTime(endtime)
        )
        
        # 根据格式选择文件扩展名
        if format.upper() == "MSEED":
            ext = ".mseed"
            write_format = "MSEED"
        elif format.upper() == "SAC":
            ext = ".sac"
            write_format = "SAC"
        elif format.upper() == "SEGY":
            ext = ".segy"
            write_format = "SEGY"
        elif format.upper() == "WAV":
            ext = ".wav"
            write_format = "WAV"
        else:
            # 默认使用MSEED
            ext = ".mseed"
            write_format = "MSEED"
        
        # 数据文件保存
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            st.write(f.name, format=write_format)
            data_path = f.name
        
        return {
            "status": "success",
            "data_file": data_path,
            "format": format.upper(),
            "traces_count": int(len(st)),  # 确保是标准Python整数
            "time_range": f"{starttime} 至 {endtime}",
            "network_station": f"{network}.{station}.{location}.{channel}",
            "message": f"成功下载 {network}.{station}.{location}.{channel} 的波形数据，格式为 {format.upper()}"
        }
    except Exception as e:
        return {"status": "error", "message": f"下载波形数据失败: {str(e)}"}


def plot_waveforms(waveform_data: str, filter_type: str = "none", freqmin: float = 0.0, freqmax: float = 0.0) -> Dict[str, Any]:
    """绘制波形数据图表
    
    Args:
        waveform_data: 格式为"network|station|location|channel|starttime|endtime"的字符串
        filter_type: 滤波类型，可选值：none, bandpass, lowpass, highpass
        freqmin: 最低频率，用于bandpass和highpass滤波
        freqmax: 最高频率，用于bandpass和lowpass滤波
        
    Returns:
        包含绘图结果信息的字典
    """
    logger.info(f"调用 plot_waveforms: {waveform_data}, 滤波: {filter_type}")
    try:
        # 解析参数
        network, station, location, channel, starttime, endtime = waveform_data.split("|")
        
        # 获取数据
        st = client.robust_call(
            "get_waveforms",
            network=network,
            station=station,
            location=location,
            channel=channel,
            starttime=UTCDateTime(starttime),
            endtime=UTCDateTime(endtime)
        )
        
        # 应用滤波器(如果指定)
        if filter_type.lower() != "none" and freqmin > 0 or freqmax > 0:
            if filter_type.lower() == "bandpass" and freqmin > 0 and freqmax > 0:
                st.filter('bandpass', freqmin=freqmin, freqmax=freqmax, corners=4)
                filter_info = f"带通滤波({freqmin}-{freqmax}Hz)"
            elif filter_type.lower() == "lowpass" and freqmax > 0:
                st.filter('lowpass', freq=freqmax, corners=4)
                filter_info = f"低通滤波(<{freqmax}Hz)"
            elif filter_type.lower() == "highpass" and freqmin > 0:
                st.filter('highpass', freq=freqmin, corners=4)
                filter_info = f"高通滤波(>{freqmin}Hz)"
            else:
                filter_info = "无滤波"
        else:
            filter_info = "无滤波"
        
        # 生成图表
        fig = st.plot(show=False, outfile=None)
        
        # 保存图片
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            fig.savefig(f.name, bbox_inches='tight')
            img_path = f.name
            
            # 在Windows下打开图片
            if os.name == 'nt':
                os.startfile(img_path)
            else:
                opener = 'open' if sys.platform == 'darwin' else 'xdg-open'
                subprocess.call([opener, img_path])
            
        return {
            "status": "success",
            "plot_path": img_path,
            "filter": filter_info,
            "traces_count": int(len(st)),  # 确保是标准Python整数
            "time_range": f"{starttime} 至 {endtime}",
            "network_station": f"{network}.{station}.{location}.{channel}",
            "message": f"成功绘制 {network}.{station}.{location}.{channel} 的波形图"
        }
    except Exception as e:
        return {"status": "error", "message": f"绘制波形图失败: {str(e)}"}


def retrieve_events(starttime: str, endtime: str, minmagnitude: float) -> Dict[str, Any]:
    """获取地震事件数据"""
    logger.info(f"调用 retrieve_events: {starttime} - {endtime}, 最小震级 {minmagnitude}")
    try:
        catalog = client.robust_call(
            "get_events",
            starttime=UTCDateTime(starttime),
            endtime=UTCDateTime(endtime),
            minmagnitude=minmagnitude
        )
        
        # 格式化事件数据
        events = []
        for event in catalog:
            origin = event.origins[0]
            magnitude = event.magnitudes[0]
            events.append({
                "time": origin.time.isoformat(),
                "magnitude": magnitude.mag,
                "type": magnitude.magnitude_type,
                "latitude": origin.latitude,
                "longitude": origin.longitude,
                "depth": origin.depth
            })

        # 不再保存数据文件，而是返回一个参数格式供下载工具使用
        catalog_data = f"{starttime}|{endtime}|{minmagnitude}"
        
        return {
            "status": "success",
            "count": len(catalog),
            "time_range": f"{starttime} 至 {endtime}",
            "min_magnitude": minmagnitude,
            "events": events,
            "catalog_data": catalog_data,  # 添加此字段以便后续下载或绘图
            "message": f"成功获取 {len(catalog)} 个地震事件"
        }
    except Exception as e:
        return {"status": "error", "message": f"获取地震事件失败: {str(e)}"}

def plot_catalog(catalog_data: str) -> Dict[str, Any]:
    """生成地震目录图表"""
    logger.info(f"调用 plot_catalog: {catalog_data}")
    try:
        starttime, endtime, minmagnitude = catalog_data.split("|")
        
        # 重新获取数据
        catalog = client.robust_call(
            "get_events",
            starttime=UTCDateTime(starttime),
            endtime=UTCDateTime(endtime),
            minmagnitude=float(minmagnitude)
        )
        
        # 生成图表
        fig = catalog.plot(show=False)
        
        # 保存图片
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            fig.savefig(f.name, bbox_inches='tight')
            img_path = f.name
            
            # 在Windows下打开图片
            if os.name == 'nt':
                os.startfile(img_path)
            else:
                opener = 'open' if sys.platform == 'darwin' else 'xdg-open'
                subprocess.call([opener, img_path])
            
        return {
            "status": "success",
            "plot_path": img_path,
            "data_identifier": catalog_data,
            "message": "图表生成成功"
        }
    except Exception as e:
        return {"status": "error", "message": f"生成图表失败: {str(e)}"}

def download_catalog_data(catalog_data: str, format: str = "QUAKEML") -> Dict[str, Any]:
    """下载地震目录数据并保存为文件
    
    Args:
        catalog_data: 格式为"starttime|endtime|minmagnitude"的字符串
        format: 输出格式，默认为QUAKEML，可选值：QUAKEML, CSV, JSON
        
    Returns:
        包含下载结果信息的字典
    """
    logger.info(f"调用 download_catalog_data: {catalog_data}, 格式: {format}")
    try:
        starttime, endtime, minmagnitude = catalog_data.split("|")
        
        # 获取数据
        catalog = client.robust_call(
            "get_events",
            starttime=UTCDateTime(starttime),
            endtime=UTCDateTime(endtime),
            minmagnitude=float(minmagnitude)
        )
        
        # 根据格式选择文件扩展名和保存方式
        if format.upper() == "QUAKEML":
            ext = ".xml"
            write_format = "QUAKEML"
        elif format.upper() == "CSV":
            ext = ".csv"
            write_format = "CSV"
        elif format.upper() == "JSON":
            ext = ".json" 
            write_format = "JSON"
        else:
            # 默认使用QUAKEML
            ext = ".xml"
            write_format = "QUAKEML"
            
        # 数据文件保存
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False, mode="w") as f:
            if format.upper() == "CSV":
                # 自定义CSV输出
                f.write("time,magnitude,magnitude_type,latitude,longitude,depth\n")
                for event in catalog:
                    origin = event.origins[0]
                    magnitude = event.magnitudes[0]
                    f.write(f"{origin.time.isoformat()},{magnitude.mag},{magnitude.magnitude_type},"
                           f"{origin.latitude},{origin.longitude},{origin.depth}\n")
            elif format.upper() == "JSON":
                # 自定义JSON输出
                events_data = []
                for event in catalog:
                    origin = event.origins[0]
                    magnitude = event.magnitudes[0]
                    events_data.append({
                        "time": origin.time.isoformat(),
                        "magnitude": magnitude.mag,
                        "magnitude_type": magnitude.magnitude_type,
                        "latitude": origin.latitude,
                        "longitude": origin.longitude,
                        "depth": origin.depth
                    })
                import json
                json.dump({"events": events_data}, f, indent=2)
            else:
                # 使用ObsPy内置的格式化器
                catalog.write(f.name, format=write_format)
                
            data_path = f.name
        
        # 返回信息
        return {
            "status": "success",
            "data_file": data_path,
            "format": format.upper(),
            "count": len(catalog),
            "time_range": f"{starttime} 至 {endtime}",
            "min_magnitude": minmagnitude,
            "message": f"成功下载 {len(catalog)} 个地震事件数据，格式为 {format.upper()}"
        }
    except Exception as e:
        return {"status": "error", "message": f"下载数据失败: {str(e)}"}

def retrieve_stations(network: str, station: str, starttime: str, endtime: str) -> Dict[str, Any]:
    """获取台站信息"""
    logger.info(f"调用 retrieve_stations: {network}.{station}")
    try:
        inventory = client.robust_call(
            "get_stations",
            network=network,
            station=station,
            starttime=UTCDateTime(starttime),
            endtime=UTCDateTime(endtime),
            level="response"
        )
        
        # 简化处理台站信息
        stations_info = []
        for net in inventory:
            for sta in net:
                channels = []
                for cha in sta:
                    channels.append({
                        "code": cha.code,
                        "location": cha.location_code,
                        "start_date": cha.start_date.isoformat() if cha.start_date else None,
                        "end_date": cha.end_date.isoformat() if cha.end_date else None,
                        "sample_rate": float(cha.sample_rate)
                    })
                
                stations_info.append({
                    "network": net.code,
                    "station": sta.code,
                    "latitude": float(sta.latitude),
                    "longitude": float(sta.longitude),
                    "elevation": float(sta.elevation),
                    "site_name": sta.site.name,
                    "creation_date": sta.creation_date.isoformat() if sta.creation_date else None,
                    "channels_count": len(channels),
                    "channels": channels[:5]  # 限制返回的通道数量
                })
        
        # 保存数据标识符供下载和绘图使用
        station_data = f"{network}|{station}|{starttime}|{endtime}"
                
        return {
            "status": "success",
            "count": len(stations_info),
            "stations": stations_info,
            "station_data": station_data,
            "time_range": f"{starttime} 至 {endtime}",
            "message": f"成功获取 {len(stations_info)} 个台站信息"
        }
    except Exception as e:
        return {"status": "error", "message": f"获取台站信息失败: {str(e)}"}

def download_stations(station_data: str, format: str = "STATIONXML") -> Dict[str, Any]:
    """下载台站数据并保存为文件
    
    Args:
        station_data: 格式为"network|station|starttime|endtime"的字符串
        format: 输出格式，默认为STATIONXML，可选值：STATIONXML, CSV, JSON
        
    Returns:
        包含下载结果信息的字典
    """
    logger.info(f"调用 download_stations: {station_data}, 格式: {format}")
    try:
        network, station, starttime, endtime = station_data.split("|")
        
        # 获取数据
        inventory = client.robust_call(
            "get_stations",
            network=network,
            station=station,
            starttime=UTCDateTime(starttime),
            endtime=UTCDateTime(endtime),
            level="response"
        )
        
        # 根据格式选择文件扩展名和保存方式
        if format.upper() == "STATIONXML":
            ext = ".xml"
            write_format = "STATIONXML"
        elif format.upper() == "CSV":
            ext = ".csv"
            write_format = "CSV"
        elif format.upper() == "JSON":
            ext = ".json" 
            write_format = "JSON"
        else:
            # 默认使用STATIONXML
            ext = ".xml"
            write_format = "STATIONXML"
            
        # 数据文件保存
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False, mode="w") as f:
            if format.upper() == "CSV":
                # 自定义CSV输出
                f.write("network,station,latitude,longitude,elevation,site_name\n")
                for net in inventory:
                    for sta in net:
                        f.write(f"{net.code},{sta.code},{sta.latitude},{sta.longitude},{sta.elevation},{sta.site.name}\n")
            elif format.upper() == "JSON":
                # 自定义JSON输出
                stations_data = []
                for net in inventory:
                    for sta in net:
                        channels = []
                        for cha in sta:
                            channels.append({
                                "code": cha.code,
                                "location": cha.location_code,
                                "start_date": cha.start_date.isoformat() if cha.start_date else None,
                                "end_date": cha.end_date.isoformat() if cha.end_date else None,
                                "sample_rate": float(cha.sample_rate)
                            })
                        
                        stations_data.append({
                            "network": net.code,
                            "station": sta.code,
                            "latitude": float(sta.latitude),
                            "longitude": float(sta.longitude),
                            "elevation": float(sta.elevation),
                            "site_name": sta.site.name,
                            "creation_date": sta.creation_date.isoformat() if sta.creation_date else None,
                            "channels": channels
                        })
                import json
                json.dump({"stations": stations_data}, f, indent=2)
            else:
                # 使用ObsPy内置的格式化器
                inventory.write(f.name, format=write_format)
                
            data_path = f.name
        
        # 返回信息
        return {
            "status": "success",
            "data_file": data_path,
            "format": format.upper(),
            "count": len(inventory),
            "time_range": f"{starttime} 至 {endtime}",
            "message": f"成功下载 {network}.{station} 的台站数据，格式为 {format.upper()}"
        }
    except Exception as e:
        return {"status": "error", "message": f"下载台站数据失败: {str(e)}"}

def plot_stations(station_data: str, map_type: str = "global") -> Dict[str, Any]:
    """绘制台站分布图
    
    Args:
        station_data: 格式为"network|station|starttime|endtime"的字符串
        map_type: 地图类型，可选值：global, regional, local
        
    Returns:
        包含绘图结果信息的字典
    """
    logger.info(f"调用 plot_stations: {station_data}, 地图类型: {map_type}")
    try:
        network, station, starttime, endtime = station_data.split("|")
        
        # 获取数据
        inventory = client.robust_call(
            "get_stations",
            network=network,
            station=station,
            starttime=UTCDateTime(starttime),
            endtime=UTCDateTime(endtime),
            level="station"
        )
        
        # 生成图表
        if map_type.lower() == "local":
            fig = inventory.plot(projection="local", show=False)
        elif map_type.lower() == "regional":
            fig = inventory.plot(projection="regional", show=False)
        else:
            fig = inventory.plot(projection="global", show=False)
        
        # 保存图片
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            fig.savefig(f.name, bbox_inches='tight', dpi=300)
            img_path = f.name
            
            # 在Windows下打开图片
            if os.name == 'nt':
                os.startfile(img_path)
            else:
                opener = 'open' if sys.platform == 'darwin' else 'xdg-open'
                subprocess.call([opener, img_path])
            
        # 统计台站数量
        station_count = sum(len(net) for net in inventory)
            
        return {
            "status": "success",
            "plot_path": img_path,
            "map_type": map_type,
            "station_count": station_count,
            "network": network,
            "time_range": f"{starttime} 至 {endtime}",
            "message": f"成功绘制 {network}.{station} 的台站分布图 ({map_type}视图)"
        }
    except Exception as e:
        return {"status": "error", "message": f"绘制台站分布图失败: {str(e)}"}

def set_client(client_type: str, data_center: str) -> Dict[str, Any]:
    """设置客户端配置"""
    logger.info(f"调用 set_client: {client_type}/{data_center}")
    return client.set_client(client_type, data_center)

def get_client_info() -> Dict[str, Any]:
    """获取当前客户端信息"""
    return client.get_current_client()

def explain_location_codes() -> Dict[str, Any]:
    """解释位置代码的含义和如何表达"""
    return {
        "status": "success",
        "message": """位置代码(location)说明:
        
        位置代码用于区分同一台站同一通道的不同传感器位置:
        - "00": 主传感器位置/主要位置/标准安装位置
        - "10": 备用传感器位置/次要位置
        - "01", "02" 等: 其他编号位置
        - 空字符串或"--": 未指定位置

        您可以使用以下自然语言表达:
        - "主传感器"、"主要位置"、"标准位置" → 系统会理解为位置代码 "00"
        - "备用传感器"、"次要位置" → 系统会理解为位置代码 "10"
        - "未指定位置"、"默认位置" → 系统会理解为空位置代码

        例如，您可以说:
        "获取IU网络ANMO台站主传感器的LHZ通道数据"
        系统会将"主传感器"解析为位置代码"00"
        """
    }
