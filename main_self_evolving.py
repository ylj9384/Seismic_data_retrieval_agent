import json, re
from z_self_evolving_test.llm_chat import generate
from z_self_evolving_test.formula_store.rag_formula import get_formula_context
from z_self_evolving_test.prompt_templates import SYSTEM_PROMPT
from z_self_evolving_test.tool_runtime import registry
from z_self_evolving_test.tool_runtime.parser import parse_action_json, register_dynamic_tool
from z_self_evolving_test.tool_runtime.sandbox import run_in_sandbox

# MAX_TURNS：保留最近 6 轮（用户+助手）上下文，避免历史无限增长
# RE_ISO：匹配 ISO 时间字符串，用于识别波形查询时间参数
MAX_TURNS = 6
RE_ISO = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")

# 构造带工具列表的系统提示
# 动态把当前工具列表注入 SYSTEM_PROMPT（tools_list 占位替换），让模型了解已有工具避免重复生成
def build_system():
    return SYSTEM_PROMPT.format(tools_list=registry.format_tools_for_prompt())

# 将系统指令、历史对话、当前用户输入串成一个单一 prompt 给 LLM（普通回答模式使用）
def build_dialog(history, user_input):
    parts = [f"[SYSTEM]\n{build_system()}"]
    for role, content in history:
        parts.append(f"[{role.upper()}]\n{content}")
    parts.append(f"[USER]\n{user_input}")
    return "\n\n".join(parts)

# 读取 action_gen 模板
# def build_action_prompt(user_query: str):
#     tmpl_path = registry.DYN_DIR.parent / "prompts" / "action_gen.txt"
#     tmpl = tmpl_path.read_text(encoding="utf-8")
#     return tmpl.format(user_query=user_query, tools=registry.format_tools_for_prompt())

def build_action_prompt(user_query: str):
    tmpl_path = registry.DYN_DIR.parent / "prompts" / "action_gen.txt"
    tmpl = tmpl_path.read_text(encoding="utf-8")
    # 拼接RAG上下文
    rag_context = get_formula_context(user_query, topk=3)
    if rag_context.strip():
        rag_part = f"\n[KNOWLEDGE]\n{rag_context}\n"
    else:
        rag_part = ""
    return tmpl.format(user_query=user_query, tools=registry.format_tools_for_prompt(), knowledge=rag_part)

# 读取 action_fix 模板
def build_fix_prompt(user_query: str, prev_code: str, error: str):
    tmpl_path = registry.DYN_DIR.parent / "prompts" / "action_fix.txt"
    tmpl = tmpl_path.read_text(encoding="utf-8")
    return tmpl.format(user_query=user_query, prev_code=prev_code, error=error)


def _normalize_result(obj):
    if isinstance(obj, dict):
        if "status" not in obj:
            obj["status"] = "success"
        return obj
    return {"status":"success","data":obj}

DEBUG_TOOL_LOG = 1
def call_tool(name: str, params: dict):
    """
    统一返回 (ok, data)
    ok=True: data 为工具原始返回对象(通常是 dict)
    ok=False: data 为错误字符串
    """
    func = registry.get(name)
    if not func:
        if DEBUG_TOOL_LOG:
            print(f"[TOOL][MISS] {name} 参数={json.dumps(params,ensure_ascii=False)}")
        return False, {"status":"error","reason":f"工具 {name} 不存在"}
    if DEBUG_TOOL_LOG:
        print(f"[TOOL][START] {name} 参数={json.dumps(params, ensure_ascii=False)}")
    try:
        raw = run_in_sandbox(func, params, timeout=15)
        norm = _normalize_result(raw)
        registry.mark_use(name, True)
        return True, norm
    except Exception as e:
        registry.mark_use(name, False)
        return False, {"status":"error","reason":str(e).splitlines()[0]}
# 工具调用结果处理与总结
def summarize_tool_output(tool_name: str, params: dict, ok: bool, data):
    """
    ok=True -> data 为 dict/list；ok=False -> data 为错误文本
    """
    if not ok:
        return f"{tool_name} 调用失败：{data}"

    # 裁剪 traces 过长
    if isinstance(data, dict):
        display = dict(data)
        if isinstance(display.get("traces"), list) and len(display["traces"]) > 5:
            display["traces_preview"] = display["traces"][:3]
            display["traces_count"] = len(display["traces"])
            del display["traces"]

        raw_json = json.dumps(display, ensure_ascii=False)

        prompt = (
            "你是地震学数据助手。下面是一次工具调用的结果，请生成规范的说明：\n"
            f"工具: {tool_name}\n"
            f"参数: {json.dumps(params or {}, ensure_ascii=False)}\n"
            f"结果JSON: {raw_json}\n"
        )
        summary = generate(prompt).strip()
        return summary

    # 非 dict/list 的成功结果（少见）直接转字符串
    return f"{tool_name} 返回：{str(data)}"

def interactive():
    history = []
    print("地学专家 自演进 Agent，输入 exit 退出。")
    while True:
        try:
            print("\n")
            print("------------------------------------------------------------")
            print("\n")
            print("已加载工具:", registry.format_tools_for_prompt())
            q = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。"); break
        if not q:
            continue
        if q.lower() in {"exit","quit"}:
            break

        # 根据问题与工具构建提示词，规定返回结果
        act_prompt = build_action_prompt(q)
        # 输入llm
        raw = generate(act_prompt)
        # 将结果中的json提取出来
        act = parse_action_json(raw) or {}

        # 处理工具调用  
        if act.get("action") == "use_tool":
            params = act.get("params") or {}
            # 在 sandbox 中执行
            ok, data = call_tool(act.get("name",""), params)
            reply = summarize_tool_output(act.get("name",""), params, ok, data)
            print("助手:", reply)
            history.append(("user", q)); history.append(("assistant", reply))
            continue

        # 提出新工具
        if act.get("action") == "propose_tool":
            name = act.get("name")
            code = act.get("code","")
            desc = act.get("desc","")
            if not (name and code):
                print("助手: propose_tool JSON 不完整，改为普通回答。")
            else:
                try:
                    register_dynamic_tool(name, code, desc)
                    print(f"助手: 新工具 {name} 已注册。")

                    # ===== 二次决策：尝试让模型现在直接 use_tool =====
                    second_prompt = build_action_prompt(q)  # 此时工具列表已包含新工具
                    raw_second = generate(second_prompt)
                    act2 = parse_action_json(raw_second) or {}

                    if act2.get("action") == "use_tool":
                        params2 = act2.get("params") or {}
                        ok2, data2 = call_tool(act2.get("name",""), params2)
                        reply2 = summarize_tool_output(act2.get("name",""), params2, ok2, data2)
                        print("助手:", reply2)
                        history.append(("user", q)); history.append(("assistant", reply2))
                        continue  # 不再走普通回答
                    else:
                        # 若再次 propose_tool 或解析失败，就放弃自动调用，继续走普通对话
                        if act2.get("action") == "propose_tool":
                            print("助手: 二次仍提出新工具，避免循环，转为普通回答。")
                except Exception as e:
                    fix_prompt = build_fix_prompt(q, code, str(e))
                    raw2 = generate(fix_prompt)
                    act_fix = parse_action_json(raw2) or {}
                    if act_fix.get("action") == "propose_tool":
                        try:
                            register_dynamic_tool(act_fix.get("name"), act_fix.get("code",""), act_fix.get("desc",""))
                            print(f"助手: 修正后工具 {act_fix.get('name')} 已注册。")
                            # 修正后同样尝试一次二次决策
                            second_prompt = build_action_prompt(q)
                            raw_second = generate(second_prompt)
                            act2 = parse_action_json(raw_second) or {}
                            if act2.get("action") == "use_tool":
                                params2 = act2.get("params") or {}
                                ok2, data2 = call_tool(act2.get("name",""), params2)
                                reply2 = summarize_tool_output(act2.get("name",""), params2, ok2, data2)
                                print("助手:", reply2)
                                history.append(("user", q)); history.append(("assistant", reply2))
                                continue
                            else:
                                if act2.get("action") == "propose_tool":
                                    print("助手: 修正后二次仍 propose_tool，转普通回答。")
                        except Exception as e2:
                            print("助手: 修正仍失败，退回文字回答。", e2)
                    else:
                        print("助手: 修正动作解析失败，退回文字回答。")

        dialog_prompt = build_dialog(history, q)
        ans = generate(dialog_prompt)
        print("助手2:", ans.strip())
        history.append(("user", q)); history.append(("assistant", ans.strip()))
        if len(history) > MAX_TURNS * 2:
            history = history[-MAX_TURNS*2:]

if __name__ == "__main__":
    interactive()