from typing import Dict, Callable, Any
from .tools import (
    detect_and_plot_phases,  # 添加这个导入
    evaluate_detection_quality,
    list_available_models, compare_models
)
from pydantic import BaseModel, Field

# 添加新的参数模型
class DetectAndPlotPhasesParams(BaseModel):
    """震相拾取并绘图参数定义"""
    waveform_file: str = Field(description="波形数据文件路径")
    model_name: str = Field(description="模型名称: PhaseNet, EQTransformer, GPD等", default="PhaseNet")
    p_threshold: float = Field(description="P波识别概率阈值", default=0.5)
    s_threshold: float = Field(description="S波识别概率阈值", default=0.5)
    detection_threshold: float = Field(description="事件检测阈值", default=0.3)
    show_probability: bool = Field(description="是否显示概率曲线", default=True)

def get_tools() -> Dict[str, Callable]:
    """
    返回震相拾取和事件检测工具函数字典
    """
    return {
        "DetectAndPlotPhases": detect_and_plot_phases,  # 新的合并工具
        "EvaluateDetectionQuality": evaluate_detection_quality,
        "ListAvailableModels": list_available_models,
        "CompareModels": compare_models,
        # 可以保留原有工具或注释掉
        # "DetectPhases": detect_phases, 
        # "PlotDetectionResult": plot_detection_result,
    }

def get_tool_descriptions() -> Dict[str, str]:
    """
    返回工具描述字典，供 LLMNode 提示词使用
    """
    return {
        "DetectAndPlotPhases": "使用深度学习模型进行震相拾取并直接绘制结果，参数：waveform_file, model_name, p_threshold, s_threshold, detection_threshold, show_probability",
        "EvaluateDetectionQuality": "评估震相拾取和事件检测质量，参数：detection_result",
        "ListAvailableModels": "列出可用的震相拾取与事件检测模型，无参数",
        "CompareModels": "比较多个模型的震相拾取结果，参数：waveform_file, models",
    }

def get_tool_param_models() -> Dict[str, Any]:
    """
    返回工具参数模型，供参数校验使用
    """
    return {
        "DetectAndPlotPhases": DetectAndPlotPhasesParams,
    }