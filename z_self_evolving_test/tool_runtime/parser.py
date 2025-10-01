import json, ast, inspect, pathlib
from typing import Optional, Dict
from . import registry
import importlib, sys, inspect

# ----------------------------------------
# 配置：仅限制根模块，放宽 from/import 细节
# ----------------------------------------
ALLOWED_ROOTS = {"obspy", "numpy", "math", "typing", "collections", "seisbench"}
FORBIDDEN_CALLS = {"exec", "eval", "open", "compile", "__import__", "system", "popen"}
MAX_FUNC_SOURCE_CHARS = 8000
MAX_FUNC_LINES = 300

def parse_action_json(text: str) -> Optional[dict]:
    """
    从模型输出中提取 JSON：
    1. 截取首个 { 到最后一个 } 之间内容
    2. 尝试 json.loads
    失败返回 None（上层自行处理）
    """
    if not text:
        return None
    s = text.strip()
    if not s.startswith("{"):
        i = s.find("{")
        j = s.rfind("}")
        if i >= 0 and j > i:
            s = s[i:j+1]
    try:
        return json.loads(s)
    except Exception:
        return None

# ---------- 安全审计辅助 ----------

def _is_allowed_import_root(module: str) -> bool:
    if not module:
        return False
    root = module.split(".")[0]
    return root in ALLOWED_ROOTS

def _check_ast(tree: ast.AST):
    """
    宽松版安全检查：
    - 允许任意 from/import，只要根模块在 ALLOWED_ROOTS
    - 禁止 from ... import *
    - 禁止 class / with / try / lambda
    - 禁止危险调用（exec/eval/open 等）
    - 仅允许一个目标函数定义（其余顶层执行语句禁止）
    """
    func_defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    others = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))]
    if len(func_defs) != 1:
        raise ValueError("代码必须且只能包含一个函数定义")
    if others:
        raise ValueError("顶层只能包含 import 与该函数定义，发现其它语句被拒绝")

    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.With, ast.Lambda,
                             ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith)):
            raise ValueError("不允许的结构（class/with/try/lambda/async）")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _is_allowed_import_root(alias.name):
                    raise ValueError(f"禁止导入模块: {alias.name} (允许: {sorted(ALLOWED_ROOTS)})")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if not _is_allowed_import_root(module):
                raise ValueError(f"禁止导入模块: {module} (允许: {sorted(ALLOWED_ROOTS)})")
            for alias in node.names:
                if alias.name == "*":
                    raise ValueError("禁止使用 from ... import *")
        elif isinstance(node, ast.Call):
            # 直接名称调用
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                raise ValueError(f"禁止调用: {node.func.id}")
            # 属性调用末尾命中危险名
            if isinstance(node.func, ast.Attribute):
                tail = node.func.attr
                if tail in FORBIDDEN_CALLS:
                    raise ValueError(f"禁止调用属性方法: {tail}")

def extract_function(code: str, func_name: str):
    """
    提取并返回 (函数对象, 函数源码片段)
    步骤：
      1. AST 解析
      2. 安全审计
      3. 精确截取函数源码
      4. 在受限命名空间 exec
    """
    norm = code.replace("\r\n", "\n")
    if len(norm) > MAX_FUNC_SOURCE_CHARS:
        raise ValueError("代码过长（字符数超限）")
    if norm.count("\n") + 1 > MAX_FUNC_LINES:
        raise ValueError("代码过长（行数超限）")

    try:
        # 生成抽象语法树 tree
        tree = ast.parse(norm)
    except SyntaxError as e:
        raise ValueError(f"语法错误: {e.msg} (line {e.lineno})") from e

    # 安全审计
    _check_ast(tree)

    target = None
    # 遍历 tree.body 找到名字等于 func_name 的 FunctionDef
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            target = node
            break
    if not target:
        raise ValueError("未找到函数定义")

    # 精确截取源码
    fn_src = ast.get_source_segment(norm, target)
    if not fn_src:
        raise ValueError("无法截取函数源码")

    # 受限执行环境：仅提供部分安全内建
    safe_builtins = {
        "len": len, "range": range, "min": min, "max": max, "sum": sum,
        "abs": abs, "float": float, "int": int, "str": str, "enumerate": enumerate,
        "zip": zip, "list": list, "dict": dict, "set": set, "sorted": sorted,
        "any": any, "all": all
    }
    # 剥离默认内建，阻断 open / exec / import 等
    glob = {"__builtins__": safe_builtins}
    loc: Dict[str, object] = {}
    # 仅执行该函数定义语句（无其它顶层副作用）
    # 把函数定义“编译并放到内存”，不做 I/O
    exec(fn_src, glob, loc)
    if func_name not in loc:
        raise ValueError("函数未正确生成")
    # 封装了该函数的代码对象 (code)、参数签名、默认值、docstring 等
    fn = loc[func_name]
    setattr(fn, "_is_dynamic_tool", True)
    return fn, fn_src

import re

# def ensure_import_math(code: str) -> str:
#     if "math." in code and "import math" not in code:
#         # 查找第一个三引号注释（支持单引号和双引号）
#         docstring_pattern = r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')'
#         match = re.search(docstring_pattern, code)
#         if match:
#             end = match.end()
#             # 在docstring后插入import math
#             return code[:end] + "\nimport math\n" + code[end:]
#         else:
#             # 没有docstring时，仍然插入到最前面
#             return "import math\n" + code
#     return code
def register_dynamic_tool(name: str, code: str, desc: str = ""):
    # code = ensure_import_math(code)
    fn, src = extract_function(code, name)  # 先做语法+安全校验
    dyn_dir = registry.DYN_DIR
    dyn_dir.mkdir(exist_ok=True)
    init_file = dyn_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("# dynamic tools package\n", encoding="utf-8")

    path = dyn_dir / f"tool_{name}.py"
    path.write_text(f"# dynamic tool: {name}\n{src}\n", encoding="utf-8")

    pkg_root = "z_self_evolving_test.dynamic_tools"
    module_name = f"{pkg_root}.tool_{name}"
    # 让 importlib 丢弃文件系统缓存，确保刚写入的 .py 文件可被发现（支持热增量）。
    importlib.invalidate_caches()
    if module_name in sys.modules:
        del sys.modules[module_name]
    module = importlib.import_module(module_name)

    if not hasattr(module, name):
        # 兜底：写文件后仍没找到（极少），把 fn 挂进去
        setattr(module, name, fn)

    real_fn = getattr(module, name)
    setattr(real_fn, "_is_dynamic_tool", True)
    setattr(real_fn, "_dynamic_module", module_name)
    setattr(real_fn, "_run_inline", True)

    sig = str(inspect.signature(real_fn))
    registry.register(name, real_fn, {
        "desc": desc,
        "signature": sig,
        "origin": "dynamic",
        "module": module_name
    })
    return real_fn