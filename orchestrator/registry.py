import logging
from typing import Dict, Any, Callable, List, Type, Optional
import importlib
import inspect

logger = logging.getLogger(__name__)

class AgentRegistry:
    """
    Agent注册中心，管理系统中所有可用的智能体
    """
    def __init__(self):
        self.agents = {}
        self.builders = {}
        self.state_classes = {}
        self.metadata = {}
        logger.info("Agent注册中心初始化")
    
    def register(self, agent_id: str, builder_func: Callable, 
                state_class: Type, keywords: List[str], 
                description: str, capabilities: List[str] = None):
        """
        注册一个新的Agent到系统中
        
        Args:
            agent_id: Agent唯一标识符
            builder_func: Agent构建函数
            state_class: Agent状态类
            keywords: 关键词列表，用于查询路由
            description: Agent描述
            capabilities: Agent能力列表
        """
        if agent_id in self.agents:
            logger.warning(f"Agent {agent_id} 已存在，将被覆盖")
        
        # 存储Agent构建函数和状态类
        self.builders[agent_id] = builder_func
        self.state_classes[agent_id] = state_class
        
        # 存储Agent元数据
        self.metadata[agent_id] = {
            "description": description,
            "keywords": keywords,
            "capabilities": capabilities or []
        }
        
        logger.info(f"Agent {agent_id} 注册成功")
    
    def build_agent(self, agent_id: str):
        """
        构建并返回指定Agent实例
        """
        if agent_id not in self.builders:
            raise ValueError(f"未知的Agent: {agent_id}")
        
        if agent_id not in self.agents:
            logger.info(f"构建Agent: {agent_id}")
            try:
                self.agents[agent_id] = self.builders[agent_id]()
                logger.info(f"Agent {agent_id} 构建成功")
            except Exception as e:
                logger.error(f"构建Agent {agent_id} 失败: {e}")
                raise
        
        return self.agents[agent_id]
    
    def get_state_class(self, agent_id: str):
        """获取Agent状态类"""
        if agent_id not in self.state_classes:
            raise ValueError(f"未知的Agent: {agent_id}")
        return self.state_classes[agent_id]
    
    def get_agent_info(self, agent_id: str = None) -> Dict:
        """
        获取Agent信息
        
        Args:
            agent_id: 指定Agent ID，为None时返回所有Agent信息
            
        Returns:
            Agent信息字典
        """
        if agent_id is not None:
            if agent_id not in self.metadata:
                raise ValueError(f"未知的Agent: {agent_id}")
            return {
                "id": agent_id,
                **self.metadata[agent_id]
            }
        else:
            return {
                agent_id: {
                    "id": agent_id,
                    **metadata
                }
                for agent_id, metadata in self.metadata.items()
            }
    
    def list_agents(self) -> List[str]:
        """列出所有注册的Agent IDs"""
        return list(self.metadata.keys())
    
    def get_agents_by_capability(self, capability: str) -> List[str]:
        """根据能力查找Agent"""
        return [
            agent_id for agent_id, metadata in self.metadata.items()
            if capability in metadata.get("capabilities", [])
        ]
    
    def discover_and_register_agents(self, base_module: str = None):
        """
        自动发现并注册Agent模块
        
        Args:
            base_module: 基础模块路径，如"data_retrieval_agent"
        """
        if base_module is None:
            logger.warning("未指定基础模块路径，跳过自动发现")
            return
        
        try:
            # 尝试导入已知的Agent模块
            known_modules = [
                f"{base_module}.agent",
                f"{base_module}.phase_detection"
            ]
            
            for module_path in known_modules:
                try:
                    module = importlib.import_module(module_path)
                    
                    # 查找build_agent函数
                    if hasattr(module, "build_agent"):
                        builder = module.build_agent
                        # 尝试从模块中获取必要的注册信息
                        state_class = getattr(module, "AgentState", None)
                        agent_id = module_path.split(".")[-1]
                        
                        if state_class:
                            self.register(
                                agent_id=agent_id,
                                builder_func=builder,
                                state_class=state_class,
                                keywords=getattr(module, "KEYWORDS", []),
                                description=getattr(module, "__doc__", "未提供描述"),
                                capabilities=getattr(module, "CAPABILITIES", [])
                            )
                            logger.info(f"自动发现并注册Agent: {agent_id}")
                except ImportError:
                    logger.info(f"模块 {module_path} 不存在，跳过")
                except Exception as e:
                    logger.warning(f"注册模块 {module_path} 失败: {e}")
        
        except Exception as e:
            logger.error(f"自动发现Agent失败: {e}")

# 创建全局注册表实例
registry = AgentRegistry()

def register_default_agents():
    """注册默认的Agent"""
    # 导入数据检索Agent
    from data_retrieval.agent_initializer import build_agent as build_data_agent
    from data_retrieval.state import AgentState as DataAgentState
    
    # 注册数据检索Agent
    registry.register(
        agent_id="data_retrieval",
        builder_func=build_data_agent,
        state_class=DataAgentState,
        keywords=["检索", "获取", "下载", "数据", "波形", "地震", "事件", "目录", 
                 "台站", "地图", "网络", "通道", "震级", "时间", "位置", "FDSN"],
        description="地震数据检索Agent，用于获取地震波形、事件和台站数据",
        capabilities=["数据获取", "波形检索", "事件查询", "台站信息", "数据可视化"]
    )
    
    # 如果震相拾取Agent已经实现，也注册它
    try:
        from phase_detection.agent_initializer import build_agent as build_phase_agent
        from phase_detection.state import PhaseDetectionState
        
        registry.register(
            agent_id="phase_detection",
            builder_func=build_phase_agent,
            state_class=PhaseDetectionState,
            keywords=["震相", "拾取", "检测", "识别", "P波", "S波", "到时", "模型", 
                     "PhaseNet", "EQTransformer", "GPD", "深度学习", "AI", "人工智能",
                     "概率", "置信度", "分析", "对比", "评估"],
            description="震相拾取与事件检测Agent，用于分析波形数据，执行震相拾取和事件检测",
            capabilities=["震相拾取", "事件检测", "P波识别", "S波识别", "波形分析"]
        )
        logger.info("震相拾取Agent注册成功")
    except ImportError:
        logger.info("震相拾取Agent模块未找到，跳过注册")
    except Exception as e:
        logger.error(f"注册震相拾取Agent失败: {e}")

# 注册默认Agent
register_default_agents()