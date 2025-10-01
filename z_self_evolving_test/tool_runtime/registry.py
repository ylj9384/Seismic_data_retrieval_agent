import json, pathlib, importlib.util, inspect
from typing import Dict, Callable, List

# 目录与文件
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DYN_DIR = BASE_DIR / "dynamic_tools"        # 动态工具源码目录
META_FILE = DYN_DIR / "tools_meta.json"     # 元数据持久化

# 内存注册表
_tool_registry: Dict[str, Callable] = {}
_tool_meta: Dict[str, dict] = {}

# ---------- 基础目录/元数据 ----------
# 确保动态工具目录 dynamic_tools 存在，不存在则创建。
def _ensure_dirs():
    DYN_DIR.mkdir(exist_ok=True)

# 读取 tools_meta.json（若存在）到内存字典 _tool_meta，用于恢复历史工具元数据（使用次数等）
def _load_meta():
    _ensure_dirs()
    if META_FILE.exists():
        try:
            data = json.loads(META_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                _tool_meta.update(data)
        except Exception:
            pass  # 元数据损坏时忽略

# 将内存中的 _tool_meta 序列化写回 tools_meta.json，持久化最新元数据。
def _save_meta():
    META_FILE.write_text(json.dumps(_tool_meta, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------- 核心操作 ----------

# 把一个工具函数登记进运行时：
# 保存到 _tool_registry[name] = func
# 合并/补全其元数据（desc/signature/origin/uses/success）进入 _tool_meta
# 立即调用 _save_meta() 持久化。
def register(name: str, func: Callable, meta: dict):
    """
    注册一个工具函数。
    meta 可包含: desc, signature, origin (builtin/dynamic)
    """
    _tool_registry[name] = func
    m = _tool_meta.get(name, {})
    if meta:
        m.update(meta)
    m.setdefault("name", name)
    m.setdefault("desc", "")
    m.setdefault("signature", "")
    m.setdefault("origin", "unknown")
    m.setdefault("uses", 0)
    m.setdefault("success", 0)
    _tool_meta[name] = m
    _save_meta()

# 在一次工具调用结束后更新统计：uses 自增；若 ok=True 则 success 自增；再持久化
def mark_use(name: str, ok: bool):
    """
    工具调用结束后更新统计。
    """
    m = _tool_meta.get(name)
    if not m:
        return
    m["uses"] = m.get("uses", 0) + 1
    if ok:
        m["success"] = m.get("success", 0) + 1
    _save_meta()

# 按名称从 _tool_registry 取回已注册的可调用函数；不存在返回 None
def get(name: str) -> Callable:
    return _tool_registry.get(name)

# 把 _tool_meta 转成列表（已归一字段），方便外部展示或调试
def list_meta() -> List[dict]:
    return [
        {
            "name": m["name"],
            "desc": m.get("desc", ""),
            "signature": m.get("signature", ""),
            "uses": m.get("uses", 0),
            "success": m.get("success", 0),
            "origin": m.get("origin", "")
        }
        for m in sorted(_tool_meta.values(), key=lambda x: x["name"])
    ]

# 将当前全部工具的元数据格式化为多行文本（name(signature) - desc uses=… ok=…），供注入到 System Prompt；若没有返回“(无已注册工具)”。
def format_tools_for_prompt() -> str:
    """
    将工具列表格式化为多行文本，注入到 System Prompt。
    """
    items = []
    for m in list_meta():
        items.append(f"{m['name']}{m['signature']} - {m['desc']} uses={m['uses']} ok={m['success']}")
    return "\n".join(items) if items else "(无已注册工具)"

# ---------- 动态工具加载 ----------

# 启动时扫描 dynamic_tools 目录下所有 tool*.py：
# 动态 import
# 找出带 _is_dynamic_tool=True 的函数
# 提取签名并调用 register() 重新注册（恢复动态工具可用状态）。
# ...existing code...
import importlib, sys, json, inspect, traceback, pathlib

DYN_DIR = pathlib.Path(__file__).parent.parent / "dynamic_tools"
META_FILE = DYN_DIR / "tools_meta.json"
BUILTIN_TOOL_NAMES = {"fetch_waveforms"}

def _load_dynamic_tools():
    """
    启动时加载 dynamic_tools 下的 tool_*.py
    - 仅依据真实文件
    - 忽略 meta 中无文件的残留条目
    - 失败不中断其它加载
    - 清理掉孤儿 meta 记录
    """
    if not DYN_DIR.exists():
        return
    # 收集实际存在的文件名集合
    file_names = {}
    for f in DYN_DIR.glob("tool_*.py"):
        name = f.stem.replace("tool_", "")
        file_names[name] = f

    # 读取已有 meta
    meta = {}
    if META_FILE.exists():
        try:
            meta = json.loads(META_FILE.read_text(encoding="utf-8") or "{}")
        except Exception:
            meta = {}

    # 需要加载的名字 = 真实文件名集合（不包含内置）
    to_load = [n for n in file_names.keys() if n not in BUILTIN_TOOL_NAMES]

    pkg_root = "z_self_evolving_test.dynamic_tools"
    loaded = {}
    for name in to_load:
        mod_name = f"{pkg_root}.tool_{name}"
        try:
            importlib.invalidate_caches()
            if mod_name in sys.modules:
                del sys.modules[mod_name]
            module = importlib.import_module(mod_name)
            if not hasattr(module, name):
                # 函数名与文件不匹配，跳过
                continue
            fn = getattr(module, name)
            if name in _tool_registry:
                continue
            setattr(fn, "_is_dynamic_tool", True)
            setattr(fn, "_run_inline", True)
            register(name, fn, {
                "desc": getattr(fn, "__doc__", "").strip().splitlines()[0][:120] if fn.__doc__ else "",
                "signature": str(inspect.signature(fn)),
                "origin": "dynamic",
                "module": mod_name
            })
            loaded[name] = True
        except Exception:
            # 打印调试但不中断
            traceback.print_exc()

    # 清理 meta 中已不存在的条目（孤儿：不在文件系统里）
    orphan_keys = [k for k in meta.keys() if k not in file_names]
    changed = False
    for k in orphan_keys:
        del meta[k]
        changed = True
    # 更新已加载工具的基本统计（如果 meta 没有）
    for k in loaded.keys():
        if k not in meta:
            meta[k] = {"uses": 0, "ok": 0}
            changed = True

    if changed:
        try:
            META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass


# 在文件末尾或 register 定义后添加：
_load_dynamic_tools()
# ---------- 启动初始化 ----------

# 启动初始化总控：
# 先 _load_meta() 恢复历史元数据
# 尝试导入并注册内置工具 fetch_waveforms
# 再加载动态工具 _load_dynamic_tools()
# 模块末尾调用 bootstrap() 使导入即完成初始化。
def bootstrap():
    _load_meta()
    # 注册内置工具（延迟导入防循环）
    try:
        from .builtin_tools import fetch_waveforms
        register("fetch_waveforms", fetch_waveforms, {
            "desc": "获取地震波形摘要(network,station,start,end,channel, location)",
            "signature": "(network, station, start, end, channel, location)",
            "origin": "builtin"
        })
    except Exception:
        pass
    _load_dynamic_tools()

# 模块导入时立即执行
bootstrap()