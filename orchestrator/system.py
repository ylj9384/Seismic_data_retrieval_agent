import logging
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from .registry import registry, register_default_agents
from .agent_supervisor import AgentSupervisorOrchestrator
from typing import Dict, Any
from orchestrator.schemas import OrchestratorResult

logger = logging.getLogger(__name__)

class OrchestratorSystem:
    """多智能体编排系统入口"""
    
    def __init__(self, llm=None, mode: str = "supervisor"):
        # 初始化LLM
        self.llm = llm
        
        # 确保注册默认Agent
        register_default_agents()
        
        # 初始化编排器
        if mode.lower() == "supervisor":
            logger.info("使用Agent Supervisor编排器")
            self.orchestrator = AgentSupervisorOrchestrator(self.llm)
        else:
            raise ValueError(f"不支持的编排模式: {mode}")
    
    def process(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """处理用户查询"""
        return self.orchestrator.orchestrate(query, context)
    
    def get_agent_info(self) -> Dict[str, Dict[str, Any]]:
        """获取所有Agent信息"""
        return {
            agent_id: {
                "description": info.get("description", ""),
                "capabilities": info.get("capabilities", [])
            }
            for agent_id, info in registry.metadata.items()
        }
