import os
import json
import numpy as np
import faiss
import dashscope

CUR_DIR = os.path.dirname(__file__)
INDEX_FILE = os.path.join(CUR_DIR, "pga_formula_index.faiss")
META_FILE = os.path.join(CUR_DIR, "pga_formula_meta.jsonl")
EMBED_DIM = 1536

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "你的Qwen-API-KEY")

def qwen_embedding(text, model="text-embedding-v1"):
    dashscope.api_key = DASHSCOPE_API_KEY
    rsp = dashscope.TextEmbedding.call(model=model, input=text)
    if rsp.status_code == 200:
        return rsp.output["embeddings"][0]["embedding"]
    else:
        print("Qwen embedding error:", rsp.message)
        return [0.0] * EMBED_DIM

_index = None
_meta = None

def _load_index():
    global _index
    if _index is None:
        _index = faiss.read_index(INDEX_FILE)
    return _index

def _load_meta():
    global _meta
    if _meta is None:
        with open(META_FILE, "r", encoding="utf-8") as f:
            _meta = [json.loads(line) for line in f]
    return _meta

def build_embedding_text(rec):
    # 参数名与含义拼接
    param_str = ""
    params = rec.get("parameters", {})
    if isinstance(params, dict):
        param_str = " ".join([f"{k}:{v}" for k, v in params.items()])
    # 其他关键信息拼接
    return (
        f"model_id={rec.get('model_id','')} "
        f"author_year={rec.get('author_year','')} "
        f"equation_desc={rec.get('equation_desc','')} "
        f"application_scope={rec.get('application_scope','')} "
        f"equation={rec.get('equation','')} "
        f"parameters={param_str}"
    )

def search_formula(query, topk=3):
    index = _load_index()
    meta = _load_meta()
    q_emb = np.array(qwen_embedding(query), dtype="float32").reshape(1, -1)
    D, I = index.search(q_emb, topk)
    results = []
    for idx, dist in zip(I[0], D[0]):
        if idx < 0 or idx >= len(meta): continue
        rec = meta[idx]
        results.append({"dist": float(dist), "record": rec})
    return results

def build_rag_context(query, topk=3):
    hits = search_formula(query, topk=topk)
    context = "\n\n".join(
        f"【模型】{h['record']['model']}【类型】{h['record']['type']}【公式】{h['record']['equation']}\n【参考】{h['record']['reference']}"
        for h in hits
    )
    print(f"[RAG] 找到 {len(hits)} 条可能相关公式: \n{context}")
    return context

def get_formula_context(query, topk=3):
    return build_rag_context(query, topk=topk)