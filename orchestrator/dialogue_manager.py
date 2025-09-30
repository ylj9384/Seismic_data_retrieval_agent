class DialogueManager:
    """
    管理多轮对话上下文和参数补全。
    """
    def __init__(self):
        self.history = []  # 存储对话历史
        self.context = {}  # 当前参数上下文

    def add_user_input(self, user_input: str):
        self.history.append({"role": "user", "content": user_input})

    def add_agent_output(self, agent_output: str):
        self.history.append({"role": "agent", "content": agent_output})

    def update_context(self, params: dict):
        """
        用用户补充的参数更新上下文
        """
        for k, v in params.items():
            if v is not None:
                self.context[k] = v

    def get_context(self):
        return self.context

    def get_history(self):
        return self.history

    def clear(self):
        self.history = []
        self.context = {}

# 单例模式（可选）
dialogue_manager = DialogueManager()