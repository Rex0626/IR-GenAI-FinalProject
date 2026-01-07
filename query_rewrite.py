# query_rewrite.py
import json
import re
import jieba

PROMPT_A = """把使用者問題改寫成 3 個關鍵字查詢，適合用在搜尋引擎。
請只輸出 JSON，格式如下：
{{"queries": ["q1", "q2", "q3"]}}

使用者問題：{question}
"""

PROMPT_B = """你是一位資訊檢索（Information Retrieval）助理。
任務：把使用者的自然語言問題，轉換成 1–3 條「適合 TF-IDF 關鍵字搜尋」的查詢字串。

規則：
- 避免產生過於泛的查詢（例如只寫「名單」「集訓」「選手」）。每條 query 必須至少包含 2 個問題中的核心詞（如：中華隊/棒球/集訓/名單）。
- 若問題包含特定運動（例如「棒球」），每條 query 都應包含該運動詞，避免撈到其他運動。
- 只能輸出「合法 JSON」，不要輸出任何其他文字（不要解釋、不要加 Markdown）。
- JSON 格式必須完全符合：{{"queries": ["q1", "q2", "q3"]}}
- 每條 query 必須是簡短的「關鍵字片語」，不要寫成完整句子，也不要加問號。
- 盡量避免只輸出 1 個詞；每條 query 建議 2–6 個詞，讓查詢更具體。
- 不要回答問題；不要新增使用者沒提到的事實或細節。
- 如果問題太模糊，請產生較廣泛但仍相關的關鍵字查詢。
- 請輸出「繁體中文」關鍵字查詢（除非專有名詞必須用英文，例如 NBA、MLB、WBC）。
- 每條 query 至少包含 1 個繁體中文詞（例如：新聞、最新、戰績、交易、傷兵、賽程…），不要整句都英文。

使用者問題：{question}
"""

import requests

OLLAMA_MODEL = "gemma2:2b"  # 你 pull 的模型

def call_llm(prompt: str) -> str:
    r = requests.post(
        "http://127.0.0.1:11434/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": "Return ONLY valid JSON."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.2},
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]

def _extract_json(text: str) -> dict:
    """
    盡量把模型輸出轉成 JSON（避免它多吐一些字導致解析失敗）
    """
    # 找第一個 {...} 區塊
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return {"queries": []}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"queries": []}

_cache = {}

STOP = set(["的","有","是","在","和","與","嗎","哪些","什麼","如何","為何","請問","最近","整理","相關","資訊","有人","有哪些","名單"])
SPORT_TERMS = ["棒球","籃球","足球","羽球","排球","網球","桌球","擊劍","游泳","田徑","柔道","跆拳道","棒協","足協"]

def sanitize_queries(question: str, queries: list[str]) -> list[str]:
    q = question.strip()

    # 1) 偵測題目裡是否有「運動限定詞」（有的話，每條 query 都必須包含它）
    must_terms = [t for t in SPORT_TERMS if t in q]

    # 2) 取題目的核心詞（用來要求 query 至少命中 2 個核心詞，避免只剩「名單/集訓」）
    core_terms = [t.strip() for t in jieba.cut(q) if t.strip()]
    core_terms = [t for t in core_terms if t not in STOP and len(t) >= 2]
    core_terms = list(dict.fromkeys(core_terms))[:8]  # 去重保序

    cleaned = []
    for s in queries:
        if not isinstance(s, str):
            continue
        s = " ".join(s.split())  # normalize 空白
        if not s:
            continue

        # (A) 若題目有運動詞，query 必須包含該運動詞
        if must_terms and not all(mt in s for mt in must_terms):
            continue

        # (B) 避免太短/太泛：至少命中 2 個核心詞（題目太短就放寬）
        hit = sum(1 for t in core_terms if t in s)
        if len(core_terms) >= 3 and hit < 2:
            continue

        cleaned.append(s)

    # 去重
    dedup = []
    seen = set()
    for x in cleaned:
        if x not in seen:
            dedup.append(x); seen.add(x)

    # 若全部被濾光，給一個保底 query（用題目的核心詞拼一條）
    if not dedup:
        fallback = " ".join(core_terms[:4]) if core_terms else q
        dedup = [fallback]

    return dedup[:3]

def rewrite_query(question: str, version: str = "B") -> dict:
    key = (version.upper(), question.strip())
    if key in _cache:
        return _cache[key]

    prompt = (PROMPT_A if version.upper() == "A" else PROMPT_B).format(
        question=question.strip()
    )

    out = call_llm(prompt)
    data = _extract_json(out)

    queries = data.get("queries", [])
    queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
    queries = sanitize_queries(question, queries)
    queries = queries[:3]  # 1–3 條

    result = {"queries": queries}
    _cache[key] = result
    return result