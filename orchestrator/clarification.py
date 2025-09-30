def generate_clarification_prompt(missing_params, context=None, tool_name=None):
    """
    根据缺失参数和上下文生成追问用户的提示语。
    :param missing_params: 缺失参数列表
    :param context: 可选，当前上下文信息（如已知参数、工具描述等）
    :param tool_name: 可选，当前工具/步骤名称
    :return: 追问字符串
    """
    if not missing_params:
        return "请补充必要信息。"
    param_str = "、".join(missing_params)
    tool_str = f" [{tool_name}]" if tool_name else ""
    context_str = ""
    if context:
        # 可根据实际需要拼接上下文信息
        for k, v in context.items():
            if v:
                context_str += f"\n已知 {k}: {v}"
    return f"为了完成{tool_str}操作，需要您补充以下参数：{param_str}。{context_str}".strip()