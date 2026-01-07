# pe_analysis.py
# Prompt Engineering Analysis for Term Project III
# Compare Prompt A vs Prompt B on:
#   (1) GenAI Query Understanding (query rewrite)
#   (2) Retrieval quality (top5 relevance proxy)
#   (3) Hallucination in explanation (proxy)
#
# Usage:
#   Terminal 1: ollama serve
#   Terminal 2 (venv): python -u pe_analysis.py
#
# Output:
#   - Console summary
#   - pe_results.json
#   - pe_results.csv

import time
import json
import csv
import re
import pickle
from datetime import datetime

import jieba

import query_rewrite as qr
import ai_explain as ax

# =========================
# Config
# =========================
INDEX_FILE = "index.pkl"
RUNS_PER_QUESTION = 3
PROMPT_VERSIONS = ["A", "B"]

TEST_QUESTIONS = [
    # --- 棒球：名單/集訓/交易/傷兵（你原本的痛點）---
    "中華隊棒球集訓名單有哪些人？",
    "中華隊棒球投手群近況（傷兵、調整狀況）",
    "富邦悍將 最近有什麼新聞（交易/傷兵/戰績）",
    "味全龍 新洋投 近況",

    # --- 籃球：NBA + 中文問法（測你「英文 query」問題）---
    "湖人 近期 傷兵 名單",
    "勇士 最近 交易 傳聞",

    # --- 羽球：隊伍/選手/賽事（常跟世大運混在一起）---
    "戴資穎 最近比賽結果",
    "羽球 世大運 中華隊 表現",
    "羽球混雙 中華隊 近期 戰績",

    # --- 綜合體育：模糊題（看 A/B 是否產生更好 query）---
    "最近台灣體育圈最大條的新聞是什麼？",
    "今天體育新聞有什麼值得看？",
    "運動員 罰款 爭議 相關新聞",
]

TOPK_SEARCH_PER_QUERY = 10
TOPK_FINAL = 5

STOP = set(["的","有","是","在","和","與","嗎","哪些","什麼","如何","為何","請問","最近","整理","相關","資訊","有人","有哪些","名單","一下","一下子","呢","吧","啊","嗎"])
# 你可以加更多常見廢詞

# =========================
# Helpers
# =========================
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "")

def build_prompt(version: str, question: str) -> str:
    version = version.upper()
    if version == "A":
        return qr.PROMPT_A.format(question=question.strip())
    return qr.PROMPT_B.format(question=question.strip())

def parse_json_strict(raw: str) -> tuple[bool, dict]:
    s = (raw or "").strip()
    if not (s.startswith("{") and s.endswith("}")):
        return False, {"queries": []}
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "queries" in obj:
            return True, obj
        return False, {"queries": []}
    except json.JSONDecodeError:
        return False, {"queries": []}

def normalize_queries(obj: dict) -> list[str]:
    qs = obj.get("queries", [])
    if not isinstance(qs, list):
        return []
    out = []
    for q in qs:
        if isinstance(q, str) and q.strip():
            out.append(q.strip())
    return out

def rule_violations(queries: list[str]) -> list[str]:
    v = []
    for q in queries:
        if "?" in q or "？" in q:
            v.append("contains_question_mark")
            break

    sentence_markers = ["因為", "所以", "如何", "為什麼", "請問", "可以", "能不能"]
    for q in queries:
        if len(q) >= 18 and any(m in q for m in sentence_markers):
            v.append("looks_like_sentence")
            break

    for q in queries:
        if " " not in q and len(q) <= 4:
            v.append("too_short_keyword")
            break

    return v

# =========================
# Load index + simple search
# =========================
def load_index():
    with open(INDEX_FILE, "rb") as f:
        data = pickle.load(f)
    return data["index"], data["documents"]

def get_snippet(text, query_terms, length=140):
    if not query_terms:
        return (text[:length] + "...") if len(text) > length else text

    start_pos = 0
    lower = text.lower()
    for term in query_terms:
        pos = lower.find(term.lower())
        if pos != -1:
            start_pos = max(0, pos - 30)
            break

    snippet = text[start_pos:start_pos + length]
    if start_pos > 0:
        snippet = "..." + snippet
    if start_pos + length < len(text):
        snippet += "..."
    return snippet

def search_one_query(q: str, inverted_index, documents, topk=10):
    terms = [t for t in jieba.cut(q) if t.strip()]
    scores = {}
    for t in terms:
        if t in inverted_index:
            for doc_id, s in inverted_index[t]:
                scores[doc_id] = scores.get(doc_id, 0.0) + s

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]
    results = []
    for doc_id, score in ranked:
        doc = documents[doc_id]
        snippet = get_snippet(doc["text"], terms)
        results.append({
            "title": doc.get("title",""),
            "url": doc.get("url",""),
            "score": float(score),
            "snippet": snippet,
        })
    return results

def merge_results(list_of_lists, top_k=5):
    merged = {}
    for lst in list_of_lists:
        for r in lst:
            key = r["url"]
            if key not in merged or r["score"] > merged[key]["score"]:
                merged[key] = r
    ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]

# =========================
# Relevance + hallucination proxy
# =========================
def pick_key_terms(question: str, inverted_index, k=2):
    terms = [t.strip() for t in jieba.cut(question) if t.strip()]
    terms = [t for t in terms if t not in STOP and len(t) >= 2]
    # 用 df 小的當關鍵
    scored = []
    for t in dict.fromkeys(terms):
        if t in inverted_index:
            df = len(inverted_index[t])
        else:
            df = 10**9
        scored.append((df, t))
    scored.sort(key=lambda x: x[0])
    return [t for _, t in scored[:k]]

def relevance_proxy(question: str, results: list[dict], inverted_index) -> float:
    """
    很粗的相關性指標：Top5 中，有幾篇 title+snippet 命中關鍵詞(至少1個)
    """
    if not results:
        return 0.0
    keys = pick_key_terms(question, inverted_index, k=2)
    if not keys:
        return 0.0

    hit = 0
    for r in results:
        hay = (r.get("title","") + " " + strip_html(r.get("snippet",""))).lower()
        if any(k.lower() in hay for k in keys):
            hit += 1
    return hit / len(results)

def hallucination_proxy(question: str, results: list[dict], explanation: str) -> dict:
    """
    Hallucination proxy（很像你論文那種「輸出有沒有超出資源」）：
    - 把 snippets 合在一起當 evidence
    - 把 explanation 拆成 token
    - 計算 explanation 中「不在 question & 不在 evidence」的 token 比例
    """
    evidence = (question or "") + "\n" + "\n".join(
        (r.get("title","") + " " + strip_html(r.get("snippet",""))) for r in results
    )
    evidence = evidence.lower()

    # 抓「可能是資訊內容」的 token：中文2字以上 or 英數2字以上
    toks = re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z0-9]{2,}", (explanation or ""))
    toks = [t.lower() for t in toks]

    # 排除很常見的廢詞
    noise = set(["摘要","資訊","不足","相關","結果","頁面","使用者","問題","因為","所以","但是","以及","包含","顯示"])
    toks2 = [t for t in toks if t not in noise]

    if not toks2:
        return {"halluc_rate": 0.0, "ooc_examples": []}

    ooc = []
    for t in toks2:
        if t not in evidence:
            ooc.append(t)

    rate = len(ooc) / len(toks2)

    # 取前幾個當例子（去重）
    seen = set()
    examples = []
    for t in ooc:
        if t not in seen:
            examples.append(t)
            seen.add(t)
        if len(examples) >= 8:
            break

    return {"halluc_rate": round(rate, 3), "ooc_examples": examples}

# =========================
# Main experiment
# =========================
def run_once(version: str, question: str, inverted_index, documents) -> dict:
    # 清 cache：不然跑多次會看起來都一樣
    if hasattr(qr, "_cache"):
        qr._cache.clear()
    if hasattr(ax, "_cache"):
        ax._cache.clear()

    prompt = build_prompt(version, question)

    # --- Rewrite ---
    t0 = time.time()
    raw = qr.call_llm(prompt)
    rewrite_latency = time.time() - t0

    strict_ok, strict_obj = parse_json_strict(raw)
    robust_obj = qr._extract_json(raw)
    robust_queries = normalize_queries(robust_obj)[:3]
    violations = rule_violations(robust_queries)

    # --- Retrieval ---
    t1 = time.time()
    all_lists = []
    if not robust_queries:
        robust_queries = [question]
    for q2 in robust_queries:
        all_lists.append(search_one_query(q2, inverted_index, documents, topk=TOPK_SEARCH_PER_QUERY))
    merged = merge_results(all_lists, top_k=TOPK_FINAL)
    retrieval_latency = time.time() - t1

    rel = relevance_proxy(question, merged, inverted_index)

    # --- Explain (A/B) ---
    t2 = time.time()
    try:
        explanation = ax.explain_results(question, merged, version=version)
    except TypeError:
        # 如果你 ai_explain.py 還沒加 version 參數，就會進來
        explanation = ax.explain_results(question, merged)
    explain_latency = time.time() - t2

    hall = hallucination_proxy(question, merged, explanation)

    return {
        "time": now_iso(),
        "version": version.upper(),
        "question": question,

        "rewrite_latency_sec": round(rewrite_latency, 3),
        "raw": raw,
        "strict_json_ok": strict_ok,
        "robust_queries": robust_queries,
        "num_queries": len(robust_queries),
        "violations": violations,

        "retrieval_latency_sec": round(retrieval_latency, 3),
        "results_top5": merged,
        "relevance_proxy": round(rel, 3),

        "explain_latency_sec": round(explain_latency, 3),
        "explanation": explanation,
        "hallucination_proxy": hall,
    }

def summarize(rows: list[dict], version: str) -> dict:
    data = [r for r in rows if r["version"] == version.upper()]
    if not data:
        return {}

    n = len(data)
    strict_ok = sum(1 for r in data if r["strict_json_ok"])
    avg_rw = sum(r["rewrite_latency_sec"] for r in data) / n
    avg_ret = sum(r["retrieval_latency_sec"] for r in data) / n
    avg_ex = sum(r["explain_latency_sec"] for r in data) / n
    avg_q = sum(r["num_queries"] for r in data) / n
    avg_rel = sum(r["relevance_proxy"] for r in data) / n
    avg_hall = sum(r["hallucination_proxy"]["halluc_rate"] for r in data) / n

    viol_counts = {}
    for r in data:
        for v in r["violations"]:
            viol_counts[v] = viol_counts.get(v, 0) + 1

    return {
        "version": version.upper(),
        "samples": n,
        "strict_json_ok_rate": round(strict_ok / n, 3),
        "avg_num_queries": round(avg_q, 3),
        "avg_rewrite_latency_sec": round(avg_rw, 3),
        "avg_retrieval_latency_sec": round(avg_ret, 3),
        "avg_explain_latency_sec": round(avg_ex, 3),
        "avg_relevance_proxy": round(avg_rel, 3),
        "avg_hallucination_proxy": round(avg_hall, 3),
        "violations": viol_counts,
    }

def main():
    print("=== Prompt Engineering Analysis: A vs B (Rewrite + Explain) ===")
    print(f"Model: {getattr(qr, 'OLLAMA_MODEL', 'unknown')}")
    print(f"Runs per question: {RUNS_PER_QUESTION}")
    print(f"Questions: {len(TEST_QUESTIONS)}")
    print("---------------------------------------------------------------")

    inverted_index, documents = load_index()
    rows = []

    for v in PROMPT_VERSIONS:
        for q in TEST_QUESTIONS:
            for k in range(RUNS_PER_QUESTION):
                print(f"[{v}] {q} (run {k+1}/{RUNS_PER_QUESTION}) ...")
                rows.append(run_once(v, q, inverted_index, documents))

    sumA = summarize(rows, "A")
    sumB = summarize(rows, "B")

    print("\n=== Summary ===")
    print(json.dumps({"A": sumA, "B": sumB}, ensure_ascii=False, indent=2))

    with open("pe_results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    with open("pe_results.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "time","version","question",
                "rewrite_latency_sec","strict_json_ok","num_queries","robust_queries","violations",
                "retrieval_latency_sec","relevance_proxy",
                "explain_latency_sec","hallucination_proxy_rate","hallucination_ooc_examples",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow({
                "time": r["time"],
                "version": r["version"],
                "question": r["question"],

                "rewrite_latency_sec": r["rewrite_latency_sec"],
                "strict_json_ok": r["strict_json_ok"],
                "num_queries": r["num_queries"],
                "robust_queries": "|".join(r["robust_queries"]),
                "violations": "|".join(r["violations"]),

                "retrieval_latency_sec": r["retrieval_latency_sec"],
                "relevance_proxy": r["relevance_proxy"],

                "explain_latency_sec": r["explain_latency_sec"],
                "hallucination_proxy_rate": r["hallucination_proxy"]["halluc_rate"],
                "hallucination_ooc_examples": "|".join(r["hallucination_proxy"]["ooc_examples"]),
            })

    print("\nSaved: pe_results.json, pe_results.csv")
    print("Tip: 報告可以放：strict_json_ok_rate、avg_relevance_proxy、avg_hallucination_proxy 三個指標對照。")

if __name__ == "__main__":
    main()