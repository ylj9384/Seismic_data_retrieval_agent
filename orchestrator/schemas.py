from typing import List, Dict, Any, TypedDict

class PlanStep(TypedDict, total=False):
    step_id: str
    title: str
    agent_id: str
    tool: str
    params: Dict[str, Any]
    depends_on: List[str]
    code: str

class Plan(TypedDict, total=False):
    query: str
    context: Dict[str, Any]
    steps: List[PlanStep]

class OrchestratorResult(TypedDict, total=False):
    plan: Plan
    # 预留：execution, artifacts, final_answer, errors