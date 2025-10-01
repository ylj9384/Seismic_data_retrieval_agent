import multiprocessing as mp
import traceback
from typing import Callable, Dict, Any, Tuple
import traceback, multiprocessing, sys
import multiprocessing, sys, traceback
# 子进程执行的工作函数
def _worker(q: mp.Queue, func: Callable, kwargs: Dict[str, Any]):
    try:
        result = func(**kwargs)
        q.put(("ok", result))
    except Exception as e:
        q.put((
            "err",
            f"{repr(e)}\n{traceback.format_exc(limit=6)}"
        ))

def run_in_sandbox(func, params: dict, timeout: int = 15):
    """
    - 动态工具(_run_inline=True)直接本进程执行，避免 Windows spawn pickle 问题
    - 其它（内置）工具用子进程隔离（可选）
    """
    # 动态工具：直接执行
    # Windows 下 multiprocessing spawn 需要可 pickle 的函数对象，动态生成/热加载函数可能不便序列化。
    # if getattr(func, "_run_inline", False):
    return func(**params)

    # （可选）内置工具仍走子进程
    # 独立进程，崩溃/内存泄漏相对隔离（但当前实现只做超时终止，仍无资源硬限制）
    def _target(q, mod_name, fn_name, p):
        try:
            f = func
            if mod_name and mod_name in sys.modules:
                f = getattr(sys.modules[mod_name], fn_name, func)
            res = f(**p)
            q.put(("ok", res))
        except Exception as e:
            q.put(("err", f"{type(e).__name__}: {e}"))

    mod_name = getattr(func, "__module__", None)
    fn_name = getattr(func, "__name__", None)
    # 作为返回信道
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_target, args=(q, mod_name, fn_name, params))
    # p.daemon=True（主进程退出时子进程被杀）
    p.daemon = True
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        raise TimeoutError("执行超时")
    if not q.empty():
        status, payload = q.get()
        if status == "ok":
            return payload
        raise RuntimeError(payload)
    raise RuntimeError("子进程无输出")
def safe_call(func: Callable, kwargs: Dict[str, Any], timeout: int = 12) -> Tuple[bool, Any]:
    """
    包装 run_in_sandbox，返回 (success, data_or_error)
    方便调用端不用频繁写 try/except。
    """
    try:
        r = run_in_sandbox(func, kwargs, timeout=timeout)
        return True, r
    except Exception as e:
        return False, str(e)