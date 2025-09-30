import os
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field
import seisbench.models as sbm
from obspy import Stream, read, UTCDateTime

logger = logging.getLogger(__name__)

# 模型管理器 - 负责加载和管理模型
class ModelManager:
    def __init__(self):
        self.models = {}
        self.available_models = {
            "phasenet": "PhaseNet",
            "eqtransformer": "EQTransformer",
            "gpd": "GPD",
            "basicphaseae": "BasicPhaseAE"
        }
    
    def get_model(self, model_name: str):
        """获取或加载模型"""
        model_name = model_name.lower()
        if model_name not in self.available_models:
            raise ValueError(f"不支持的模型: {model_name}, 可用模型: {list(self.available_models.keys())}")
        
        if model_name not in self.models:
            # 加载模型
            logger.info(f"加载模型: {model_name}")
            if model_name == "phasenet":
                self.models[model_name] = sbm.PhaseNet.from_pretrained("original")
            elif model_name == "eqtransformer":
                self.models[model_name] = sbm.EQTransformer.from_pretrained("stead")
            elif model_name == "gpd":
                self.models[model_name] = sbm.GPD.from_pretrained("instance")
            elif model_name == "basicphaseae":
                self.models[model_name] = sbm.BasicPhaseAE.from_pretrained("stead")
        
        return self.models[model_name]

# 实例化模型管理器
model_manager = ModelManager()

def check_required_params(params: dict, required: list):
    """检查必需参数是否齐全，返回缺失项列表"""
    missing = [p for p in required if not params.get(p)]
    return missing

def detect_and_plot_phases(
    waveform_file: str, 
    model_name: str = "PhaseNet", 
    p_threshold: float = 0.5, 
    s_threshold: float = 0.5,
    detection_threshold: float = 0.3,
    show_probability: bool = True
) -> Dict[str, Any]:
    """使用深度学习模型进行震相拾取并直接绘制结果"""
    # 参数校验
    params = {"waveform_file": waveform_file}
    missing = check_required_params(params, ["waveform_file"])
    if missing:
        return {
            "clarification_needed": True,
            "missing_params": missing,
            "output": f"缺少参数：{', '.join(missing)}，请补充。"
        }
    
    import os
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    import re
    
    logger.info(f"使用{model_name}进行震相拾取并绘图: {waveform_file}")
    
    def convert_numpy_types(obj):
        """将NumPy类型转换为Python原生类型，以便JSON序列化"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    try:
        # 第1步：读取波形数据
        st = read(waveform_file)
        
        # 检查并处理单通道情况
        if len(st) == 1:
            logger.info(f"检测到单通道数据，进行通道扩展处理")
            original_trace = st[0]
            
            # 通道复制
            from obspy import Stream
            new_st = Stream()
            new_st += original_trace
            
            # 复制两个通道并修改通道名
            for suffix in ['N', 'E']:
                trace_copy = original_trace.copy()
                # 尝试保留原始通道名格式
                if len(original_trace.stats.channel) >= 3:
                    trace_copy.stats.channel = original_trace.stats.channel[:-1] + suffix
                else:
                    trace_copy.stats.channel = original_trace.stats.channel + suffix
                new_st += trace_copy
            
            st = new_st
            logger.info(f"扩展后通道: {[tr.stats.channel for tr in st]}")
        
        # 第2步：获取模型并执行震相拾取
        try:
            model = model_manager.get_model(model_name.lower())
        except ValueError as e:
            return {"status": "error", "message": str(e)}
        
        annotations = model.annotate(st)
        output = model.classify(st, P_threshold=p_threshold, S_threshold=s_threshold)
        
        # 第3步：提取震相拾取结果
        picks_result = []
        if hasattr(output, 'picks') and output.picks:
            for pick in output.picks:
                pick_info = {}
                
                # 安全提取震相类型
                if hasattr(pick, 'phase'):
                    pick_info["phase"] = pick.phase
                elif hasattr(pick, 'phase_hint'):
                    pick_info["phase"] = pick.phase_hint
                else:
                    # 如果无法确定相位，尝试推断
                    if hasattr(pick, 'trace_id'):
                        trace_id = pick.trace_id
                        if 'Z' in trace_id:
                            pick_info["phase"] = 'P'
                        elif 'N' in trace_id or 'E' in trace_id:
                            pick_info["phase"] = 'S'
                        else:
                            pick_info["phase"] = 'Unknown'
                    else:
                        pick_info["phase"] = 'Unknown'
                
                # 安全提取时间
                if hasattr(pick, 'peak_time'):
                    pick_info["time"] = pick.peak_time.isoformat()
                elif hasattr(pick, 'time'):
                    pick_info["time"] = pick.time.isoformat()
                elif hasattr(pick, 'start_time'):
                    pick_info["time"] = pick.start_time.isoformat()
                
                # 安全提取通道信息
                if hasattr(pick, 'waveform_id'):
                    waveform_id = pick.waveform_id
                    if hasattr(waveform_id, 'network_code'):
                        pick_info["network"] = waveform_id.network_code
                    if hasattr(waveform_id, 'station_code'):
                        pick_info["station"] = waveform_id.station_code
                    if hasattr(waveform_id, 'channel_code'):
                        pick_info["channel"] = waveform_id.channel_code
                    if hasattr(waveform_id, 'location_code'):
                        pick_info["location"] = waveform_id.location_code
                elif hasattr(pick, 'trace_id'):
                    pick_info["trace_id"] = pick.trace_id
                
                # 安全提取概率
                if hasattr(pick, 'probability'):
                    pick_info["probability"] = float(pick.probability)
                elif hasattr(pick, 'peak_value'):
                    pick_info["probability"] = float(pick.peak_value)
                
                picks_result.append(pick_info)
        
        # 提取事件检测结果
        detections_result = []
        if hasattr(output, 'detections') and output.detections:
            for detection in output.detections:
                det_result = {}
                
                # 安全提取开始时间
                if hasattr(detection, 'start_time'):
                    det_result["start_time"] = detection.start_time.isoformat()
                elif hasattr(detection, 'time'):
                    det_result["start_time"] = detection.time.isoformat()
                else:
                    # 尝试从字符串中提取时间
                    detection_str = str(detection)
                    time_matches = re.findall(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', detection_str)
                    if time_matches and len(time_matches) > 0:
                        det_result["start_time"] = time_matches[0]
                
                # 安全提取结束时间
                if hasattr(detection, 'end_time'):
                    det_result["end_time"] = detection.end_time.isoformat()
                elif "start_time" in det_result:
                    # 如果只有开始时间，假设持续30秒
                    start = UTCDateTime(det_result["start_time"])
                    det_result["end_time"] = (start + 30).isoformat()
                
                # 安全提取概率
                if hasattr(detection, 'peak_value'):
                    det_result["probability"] = float(detection.peak_value)
                
                # 安全提取trace_id
                if hasattr(detection, 'trace_id'):
                    det_result["trace_id"] = detection.trace_id
                
                detections_result.append(det_result)
        
        # 提取最大概率值
        probabilities = {}
        try:
            # 提取P波和S波最大概率
            if annotations.select(channel="*P"):
                p_probs = annotations.select(channel="*P")[0].data
                p_max = np.max(p_probs)
                probabilities["p_max_probability"] = float(p_max)
            
            if annotations.select(channel="*S"):
                s_probs = annotations.select(channel="*S")[0].data
                s_max = np.max(s_probs)
                probabilities["s_max_probability"] = float(s_max)
            
            # 提取事件检测最大概率
            if annotations.select(channel="*D"):
                det_probs = annotations.select(channel="*D")[0].data
                det_max = np.max(det_probs)
                probabilities["detection_max_probability"] = float(det_max)
            elif annotations.select(channel="*N"):
                noise_probs = annotations.select(channel="*N")[0].data
                det_probs = 1.0 - noise_probs
                det_max = np.max(det_probs)
                probabilities["detection_max_probability"] = float(det_max)
        except Exception as e:
            logger.error(f"提取概率值出错: {e}")
        
        # 第4步：绘制结果
        # 创建图形
        if show_probability:
            # 创建带有多个子图的图形 - 动态确定子图数量
            n_subplots = 2  # 默认：波形 + 震相概率
            if annotations.select(channel="*Detection"):
                n_subplots = 3  # 添加检测概率图
            
            fig, axs = plt.subplots(n_subplots, 1, figsize=(15, 4*n_subplots), sharex=True)
            
            # 如果只有一个子图，将axs转换为列表以便统一处理
            if n_subplots == 1:
                axs = [axs]
            
            # 1. 绘制波形
            offset = annotations[0].stats.starttime - st[0].stats.starttime
            for i in range(min(3, len(st))):
                axs[0].plot(st[i].times(), st[i].data, label=st[i].stats.channel)
            axs[0].set_title("Seismic Waveforms")
            axs[0].legend()
            
            # 2. 绘制P波和S波概率
            if annotations.select(channel="*P"):
                p_probs = annotations.select(channel="*P")[0].data
                axs[1].plot(annotations.select(channel="*P")[0].times() + offset, p_probs, 'r-', label="P-wave Probability")
            
            if annotations.select(channel="*S"):
                s_probs = annotations.select(channel="*S")[0].data
                axs[1].plot(annotations.select(channel="*S")[0].times() + offset, s_probs, 'g-', label="S-wave Probability")
            
            axs[1].set_title("Phase Probabilities")
            axs[1].axhline(p_threshold, color='red', linestyle='--', alpha=0.5)
            axs[1].axhline(s_threshold, color='green', linestyle='--', alpha=0.5)
            axs[1].legend()
            
            # 3. 绘制事件检测概率(如果有)
            if n_subplots > 2:
                if annotations.select(channel="*Detection"):
                    det_probs = annotations.select(channel="*Detection")[0].data
                    axs[2].plot(annotations.select(channel="*Detection")[0].times() + offset, det_probs, 'b-', label="Event Detection Probability")
                elif annotations.select(channel="*N"):
                    noise_probs = annotations.select(channel="*N")[0].data
                    det_probs = 1.0 - noise_probs
                    axs[2].plot(annotations.select(channel="*N")[0].times() + offset, det_probs, 'b-', label="Event Detection Probability")
                
                axs[2].set_title("Event Detection Probability")
                axs[2].axhline(detection_threshold, color='blue', linestyle='--', alpha=0.5)
                axs[2].legend()
            
            # 标记震相拾取和事件检测结果
            if hasattr(output, 'picks') and output.picks:
                for pick in output.picks:
                    # 安全获取时间和相位
                    pick_time = None
                    phase = None
                    
                    # 获取时间
                    if hasattr(pick, 'peak_time'):
                        pick_time = pick.peak_time
                    elif hasattr(pick, 'time'):
                        pick_time = pick.time
                    elif hasattr(pick, 'start_time'):
                        pick_time = pick.start_time
                        
                    # 获取相位
                    if hasattr(pick, 'phase'):
                        phase = pick.phase
                    elif hasattr(pick, 'phase_hint'):
                        phase = pick.phase_hint
                        
                    if pick_time is not None:
                        rel_time = (pick_time - st[0].stats.starttime)
                        color = 'red' if phase == 'P' else 'green'
                        
                        for ax in axs:
                            ax.axvline(rel_time, color=color, linestyle='--')
                            # 在顶部添加标签
                            ylim = ax.get_ylim()
                            ax.text(rel_time, ylim[1]*0.95, phase, color=color, 
                                   horizontalalignment='center', verticalalignment='top')
            
            # 标记事件检测结果
            if hasattr(output, 'detections') and output.detections:
                for detection in output.detections:
                    start_time = None
                    end_time = None
                    
                    # 获取开始时间
                    if hasattr(detection, 'start_time'):
                        start_time = detection.start_time
                    elif hasattr(detection, 'time'):
                        start_time = detection.time
                        
                    # 获取结束时间
                    if hasattr(detection, 'end_time'):
                        end_time = detection.end_time
                    elif start_time is not None:
                        if hasattr(detection, 'duration'):
                            end_time = start_time + detection.duration
                        else:
                            # 假设30秒的持续时间
                            end_time = start_time + 30
                    
                    if start_time is not None and end_time is not None:
                        start_rel = (start_time - st[0].stats.starttime)
                        end_rel = (end_time - st[0].stats.starttime)
                        
                        for ax in axs:
                            ax.axvspan(start_rel, end_rel, color='blue', alpha=0.1)
        else:
            # 简单绘制，只包含波形和震相标记
            fig = plt.figure(figsize=(15, 5))
            ax = fig.add_subplot(111)
            
            for i in range(min(3, len(st))):
                ax.plot(st[i].times(), st[i].data, label=st[i].stats.channel)
            
            # 标记震相拾取
            if hasattr(output, 'picks') and output.picks:
                for pick in output.picks:
                    pick_time = None
                    phase = None
                    
                    # 获取时间
                    if hasattr(pick, 'peak_time'):
                        pick_time = pick.peak_time
                    elif hasattr(pick, 'time'):
                        pick_time = pick.time
                    elif hasattr(pick, 'start_time'):
                        pick_time = pick.start_time
                        
                    # 获取相位
                    if hasattr(pick, 'phase'):
                        phase = pick.phase
                    elif hasattr(pick, 'phase_hint'):
                        phase = pick.phase_hint
                    
                    if pick_time is not None:
                        rel_time = (pick_time - st[0].stats.starttime)
                        color = 'red' if phase == 'P' else 'green'
                        label = f"{phase if phase else '未知'}波 ({pick_time.isoformat()})"
                        ax.axvline(rel_time, color=color, linestyle='--', label=label)
            
            ax.legend()
            ax.set_title("地震波形与震相拾取结果")
        
        # 保存图片
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            plt.tight_layout()
            fig.savefig(f.name, dpi=300, bbox_inches='tight')
            img_path = f.name
            
            # 在Windows下打开图片
            if os.name == 'nt':
                try:
                    os.startfile(img_path)
                except:
                    logger.warning(f"无法自动打开图像: {img_path}")
        
        # 保存数据供后续使用（如评估质量）
        import uuid
        detection_id = str(uuid.uuid4())
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump({"annotations": annotations, "output": output, "stream": st}, f)
            data_cache_path = f.name

        # 格式化震相时间信息
        p_picks = [p for p in picks_result if p.get("phase") == "P"]
        s_picks = [s for s in picks_result if s.get("phase") == "S"]
        p_time_str = "未检测到" if not p_picks else p_picks[0].get("time", "未知")
        s_time_str = "未检测到" if not s_picks else s_picks[0].get("time", "未知")

        # 格式化事件时间信息
        event_time_str = "未检测到"
        event_end_str = ""
        if detections_result:
            event_time_str = detections_result[0].get("start_time", "未知")
            if "end_time" in detections_result[0]:
                event_end_str = f"至 {detections_result[0]['end_time']}"

        # 创建详细消息
        detailed_message = f"""使用{model_name}成功完成震相拾取与绘图，找到{len(picks_result)}个震相和{len(detections_result)}个事件
        震相时间信息:
        - P波到达时间: {p_time_str}
        - S波到达时间: {s_time_str}
        - 事件时间: {event_time_str} {event_end_str}"""
        
        # 第5步：整合结果
        result = {
            "status": "success",
            "detection_id": detection_id,
            "model": model_name,
            "picks_count": len(picks_result),
            "detections_count": len(detections_result),
            "picks": picks_result,
            "detections": detections_result,
            "probabilities": probabilities,
            "plot_path": img_path,
            "data_cache": data_cache_path,
            "message": detailed_message
        }
        
        # 确保所有值都是JSON可序列化的
        result = convert_numpy_types(result)
        
        return result
    except Exception as e:
        logger.error(f"震相拾取与绘图失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"status": "error", "message": f"震相拾取与绘图失败: {str(e)}"}
def evaluate_detection_quality(detection_result: str) -> Dict[str, Any]:
    """评估震相拾取和事件检测质量
    
    Args:
        detection_result: 检测结果标识符
        
    Returns:
        包含评估结果的字典
    """
    params = {"detection_result": detection_result}
    missing = check_required_params(params, ["detection_result"])
    if missing:
        return {
            "clarification_needed": True,
            "missing_params": missing,
            "output": f"缺少参数：{', '.join(missing)}，请补充。"
        }
    
    logger.info(f"评估检测质量: {detection_result}")
    
    try:
        # 加载缓存的数据
        import pickle
        with open(detection_result, "rb") as f:
            data = pickle.load(f)
        
        annotations = data["annotations"]
        output = data["output"]
        
        # 提取震相概率
        probabilities = {}
        p_quality = "未知"
        s_quality = "未知"
        event_quality = "未知"
        
        if annotations.select(channel="*P"):
            p_probs = annotations.select(channel="*P")[0].data
            p_max = float(np.max(p_probs))
            probabilities["p_max"] = p_max
            
            # 评估P波拾取质量
            if p_max > 0.9:
                p_quality = "极好"
            elif p_max > 0.7:
                p_quality = "良好"
            elif p_max > 0.5:
                p_quality = "一般"
            else:
                p_quality = "较差"
        
        if annotations.select(channel="*S"):
            s_probs = annotations.select(channel="*S")[0].data
            s_max = float(np.max(s_probs))
            probabilities["s_max"] = s_max
            
            # 评估S波拾取质量
            if s_max > 0.9:
                s_quality = "极好"
            elif s_max > 0.7:
                s_quality = "良好"
            elif s_max > 0.5:
                s_quality = "一般"
            else:
                s_quality = "较差"
        
        # 评估事件检测质量
        if annotations.select(channel="*D"):
            det_probs = annotations.select(channel="*D")[0].data
            det_max = float(np.max(det_probs))
            probabilities["det_max"] = det_max
            
            if det_max > 0.9:
                event_quality = "极好"
            elif det_max > 0.7:
                event_quality = "良好"
            elif det_max > 0.5:
                event_quality = "一般"
            else:
                event_quality = "较差"
        elif annotations.select(channel="*N"):
            noise_probs = annotations.select(channel="*N")[0].data
            det_probs = 1.0 - noise_probs
            det_max = float(np.max(det_probs))
            probabilities["det_max"] = det_max
            
            if det_max > 0.9:
                event_quality = "极好"
            elif det_max > 0.7:
                event_quality = "良好"
            elif det_max > 0.5:
                event_quality = "一般"
            else:
                event_quality = "较差"
        
        # 返回评估结果
        return {
            "status": "success",
            "probabilities": probabilities,
            "p_wave_quality": p_quality,
            "s_wave_quality": s_quality,
            "event_detection_quality": event_quality,
            "message": f"P波拾取质量: {p_quality}, S波拾取质量: {s_quality}, 事件检测质量: {event_quality}"
        }
    except Exception as e:
        logger.error(f"评估检测质量失败: {e}")
        return {"status": "error", "message": f"评估检测质量失败: {str(e)}"}

def list_available_models() -> Dict[str, Any]:
    """列出可用的震相拾取与事件检测模型"""
    # 无必需参数，无需校验
    try:
        models_info = {
            "PhaseNet": {
                "description": "基于U-Net架构的震相拾取模型",
                "capabilities": ["P波拾取", "S波拾取"],
                "paper": "Zhu, W., & Beroza, G. C. (2019). PhaseNet: A deep-neural-network-based seismic arrival-time picking method"
            },
            "EQTransformer": {
                "description": "结合CNN、RNN和注意力机制的地震检测与拾取模型",
                "capabilities": ["P波拾取", "S波拾取", "事件检测"],
                "paper": "Mousavi, S. M., Ellsworth, W. L., Zhu, W., Chuang, L. Y., & Beroza, G. C. (2020). Earthquake transformer"
            },
            "GPD": {
                "description": "通用震相检测器，用于震相识别",
                "capabilities": ["P波拾取", "S波拾取"],
                "paper": "Ross, Z. E., Meier, M. A., & Hauksson, E. (2018). Generalized seismic phase detection with deep learning"
            },
            "BasicPhaseAE": {
                "description": "基于自编码器的震相识别模型",
                "capabilities": ["P波拾取", "S波拾取"],
                "paper": "Woollam, J., Münchmeyer, J., et al. (2022). SeisBench: A toolbox for machine learning in seismology"
            }
        }
        
        return {
            "status": "success",
            "available_models": list(models_info.keys()),
            "models_info": models_info,
            "message": "成功获取可用模型列表"
        }
    except Exception as e:
        return {"status": "error", "message": f"获取模型列表失败: {str(e)}"}

def compare_models(waveform_file: str, models: List[str] = None) -> Dict[str, Any]:
    """比较多个模型的震相拾取结果，将结果绘制到一张图上
    
    Args:
        waveform_file: 波形数据文件路径
        models: 要比较的模型列表，默认为所有可用模型
        
    Returns:
        包含比较结果的字典
    """
    # 参数校验
    params = {"waveform_file": waveform_file}
    missing = check_required_params(params, ["waveform_file"])
    if missing:
        return {
            "clarification_needed": True,
            "missing_params": missing,
            "output": f"缺少参数：{', '.join(missing)}，请补充。"
        }
    
    logger.info(f"比较模型震相拾取结果: {waveform_file}")
    
    if not models:
        models = ["PhaseNet", "EQTransformer", "BasicPhaseAE", "GPD"]

    try:
        # 第1步：读取波形数据
        st = read(waveform_file)
        
        # 检查并处理单通道情况
        if len(st) == 1:
            logger.info(f"检测到单通道数据，进行通道扩展处理")
            original_trace = st[0]
            
            # 通道复制
            from obspy import Stream
            new_st = Stream()
            new_st += original_trace
            
            # 复制两个通道并修改通道名
            for suffix in ['N', 'E']:
                trace_copy = original_trace.copy()
                # 尝试保留原始通道名格式
                if len(original_trace.stats.channel) >= 3:
                    trace_copy.stats.channel = original_trace.stats.channel[:-1] + suffix
                else:
                    trace_copy.stats.channel = original_trace.stats.channel + suffix
                new_st += trace_copy
            
            st = new_st
            logger.info(f"扩展后通道: {[tr.stats.channel for tr in st]}")
        
        # 第2步：执行各模型并收集结果
        model_results = {}
        model_annotations = {}
        
        for model_name in models:
            try:
                logger.info(f"加载模型: {model_name}")
                model = model_manager.get_model(model_name.lower())
                
                # 获取模型注释和输出
                annotations = model.annotate(st)
                output = model.classify(st, P_threshold=0.5, S_threshold=0.5)
                
                model_results[model_name] = output
                model_annotations[model_name] = annotations
                
                logger.info(f"模型 {model_name} 处理完成")
            except Exception as e:
                logger.error(f"模型 {model_name} 处理失败: {e}")
        
        # 第3步：创建比较图 - 与detect_and_plot_phases保持一致的样式
        # 创建包含多个子图的大图 - 4个模型，每个模型2行子图（波形+概率）
        n_models = len(model_results)
        if n_models == 0:
            return {"status": "error", "message": "没有成功加载任何模型"}
        
        # 创建主图和子图网格 - 每个模型2行（波形+概率）
        fig = plt.figure(figsize=(15, 5 * n_models))
        fig.suptitle("Seismic Phase Picking Comparison", fontsize=16)
        
        # 创建子图网格：每个模型2个子图（波形+概率）
        gs = fig.add_gridspec(n_models * 2, 1, hspace=0.3)
        axs = []
        
        # 遍历模型并绘制
        for i, model_name in enumerate(model_results.keys()):
            # 跳过模型注释缺失的情况
            if model_name not in model_annotations:
                continue
            
            output = model_results[model_name]
            annotations = model_annotations[model_name]
            
            # 计算时间偏移
            offset = annotations[0].stats.starttime - st[0].stats.starttime
            
            # 1. 创建波形子图
            ax_wave = fig.add_subplot(gs[i*2])
            axs.append(ax_wave)
            
            # 绘制波形
            for j in range(min(3, len(st))):
                ax_wave.plot(st[j].times(), st[j].data, label=st[j].stats.channel)
            
            ax_wave.set_title(f"{model_name} - Seismic Waveforms", fontsize=12)
            ax_wave.legend(loc='upper right')
            
            # 2. 创建概率子图
            ax_prob = fig.add_subplot(gs[i*2+1], sharex=ax_wave)
            axs.append(ax_prob)
            
            # 绘制P波和S波概率
            if annotations.select(channel="*P"):
                p_probs = annotations.select(channel="*P")[0].data
                ax_prob.plot(annotations.select(channel="*P")[0].times() + offset, p_probs, 'r-', label="P-wave Probability")
            
            if annotations.select(channel="*S"):
                s_probs = annotations.select(channel="*S")[0].data
                ax_prob.plot(annotations.select(channel="*S")[0].times() + offset, s_probs, 'g-', label="S-wave Probability")
            
            ax_prob.set_title(f"{model_name} - Phase Probabilities", fontsize=12)
            ax_prob.axhline(0.5, color='red', linestyle='--', alpha=0.5)
            ax_prob.axhline(0.5, color='green', linestyle='--', alpha=0.5)
            ax_prob.legend(loc='upper right')
            
            # 仅为最后一个模型添加x轴标签
            if i == n_models - 1:
                ax_prob.set_xlabel("Time (seconds)")
            
            # 标记震相拾取结果
            if hasattr(output, 'picks') and output.picks:
                for pick in output.picks:
                    # 安全获取时间和相位
                    pick_time = None
                    phase = None
                    
                    # 获取时间
                    if hasattr(pick, 'peak_time'):
                        pick_time = pick.peak_time
                    elif hasattr(pick, 'time'):
                        pick_time = pick.time
                    elif hasattr(pick, 'start_time'):
                        pick_time = pick.start_time
                        
                    # 获取相位
                    if hasattr(pick, 'phase'):
                        phase = pick.phase
                    elif hasattr(pick, 'phase_hint'):
                        phase = pick.phase_hint
                        
                    if pick_time is not None:
                        rel_time = (pick_time - st[0].stats.starttime)
                        color = 'red' if phase == 'P' else 'green'
                        
                        # 在两个子图上都标记震相线
                        for ax in [ax_wave, ax_prob]:
                            ax.axvline(rel_time, color=color, linestyle='--')
                            # 在顶部添加标签
                            ylim = ax.get_ylim()
                            ax.text(rel_time, ylim[1]*0.95, phase, color=color, 
                                    horizontalalignment='center', verticalalignment='top')
        
        # 调整布局
        plt.tight_layout()
        fig.subplots_adjust(top=0.95)  # 为总标题留出空间
        
        # 保存图片
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            fig.savefig(f.name, dpi=300, bbox_inches='tight')
            img_path = f.name
            
            # 在Windows下打开图片
            if os.name == 'nt':
                try:
                    os.startfile(img_path)
                except:
                    logger.warning(f"无法自动打开图像: {img_path}")
        
        # 收集各模型的震相拾取结果用于返回
        comparison_results = {}
        for model_name in model_results:
            output = model_results[model_name]
            annotations = model_annotations[model_name]
            
            # 提取震相拾取结果
            picks_list = []
            if hasattr(output, 'picks') and output.picks:
                for pick in output.picks:
                    pick_info = {}
                    
                    # 安全提取震相类型
                    if hasattr(pick, 'phase'):
                        pick_info["phase"] = pick.phase
                    elif hasattr(pick, 'phase_hint'):
                        pick_info["phase"] = pick.phase_hint
                    else:
                        # 如果无法确定相位，尝试推断
                        if hasattr(pick, 'trace_id'):
                            trace_id = pick.trace_id
                            if 'Z' in trace_id:
                                pick_info["phase"] = 'P'
                            elif 'N' in trace_id or 'E' in trace_id:
                                pick_info["phase"] = 'S'
                            else:
                                pick_info["phase"] = 'Unknown'
                        else:
                            pick_info["phase"] = 'Unknown'
                    
                    # 安全提取时间
                    if hasattr(pick, 'peak_time'):
                        pick_info["time"] = pick.peak_time.isoformat()
                    elif hasattr(pick, 'time'):
                        pick_info["time"] = pick.time.isoformat()
                    elif hasattr(pick, 'start_time'):
                        pick_info["time"] = pick.start_time.isoformat()
                    
                    # 安全提取概率
                    if hasattr(pick, 'probability'):
                        pick_info["probability"] = float(pick.probability)
                    elif hasattr(pick, 'peak_value'):
                        pick_info["probability"] = float(pick.peak_value)
                    
                    picks_list.append(pick_info)
            
            # 提取最大概率值
            probabilities = {}
            try:
                # 提取P波和S波最大概率
                if annotations.select(channel="*P"):
                    p_probs = annotations.select(channel="*P")[0].data
                    p_max = np.max(p_probs)
                    probabilities["p_max_probability"] = float(p_max)
                
                if annotations.select(channel="*S"):
                    s_probs = annotations.select(channel="*S")[0].data
                    s_max = np.max(s_probs)
                    probabilities["s_max_probability"] = float(s_max)
            except Exception as e:
                logger.error(f"提取{model_name}概率值出错: {e}")
            
            # 将此模型的结果添加到比较结果中
            comparison_results[model_name] = {
                "picks_count": len(picks_list),
                "picks": picks_list,
                "probabilities": probabilities
            }
        
        # 创建结果摘要
        summary = []
        for model_name, results in comparison_results.items():
            p_picks = [p for p in results["picks"] if p.get("phase") == "P"]
            s_picks = [s for s in results["picks"] if s.get("phase") == "S"]
            
            p_time = "Not detected" if not p_picks else p_picks[0].get("time", "Unknown")
            s_time = "Not detected" if not s_picks else s_picks[0].get("time", "Unknown")
            
            model_summary = f"{model_name}:\n"
            model_summary += f"- Detected {len(p_picks)} P-waves (Time: {p_time})\n"
            model_summary += f"- Detected {len(s_picks)} S-waves (Time: {s_time})\n"
            
            if "probabilities" in results:
                probs = results["probabilities"]
                if "p_max_probability" in probs:
                    model_summary += f"- P-wave max probability: {probs['p_max_probability']:.3f}\n"
                if "s_max_probability" in probs:
                    model_summary += f"- S-wave max probability: {probs['s_max_probability']:.3f}\n"
            
            summary.append(model_summary)
        
        # 返回结果
        return {
            "status": "success",
            "plot_path": img_path,
            "comparison_results": comparison_results,
            "summary": "\n".join(summary),
            "message": f"Successfully compared phase picking results from {len(model_results)} models",
            "waveform_file": waveform_file
        }
    except Exception as e:
        logger.error(f"比较模型结果失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"status": "error", "message": f"比较模型结果失败: {str(e)}"}