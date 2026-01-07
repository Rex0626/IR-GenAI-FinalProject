# ai_explain.py
import requests
import re

OLLAMA_MODEL = "gemma2:2b"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

PROMPT_EXPLAIN_A = """你是一位體育新聞助理。
任務：根據 snippets 直接回答使用者問題，並用 3–5 句話說明。

規則（寬鬆版，可能會推測）：
- 請盡量給出完整答案，即使 snippets 不足，也可以做合理推測讓回答完整。
- 文字要自然，不必一直說「資訊不足」。
- 不用每句都引用編號，但可以偶爾引用如：[1]。

使用者問題：
{question}

提供的 snippets：
{snippets_block}
"""

PROMPT_EXPLAIN_B = """你是一位資訊檢索（Information Retrieval）助理。
任務：根據「提供的搜尋結果摘要 snippets」，解釋哪些頁面與使用者問題相關，以及為何相關。

嚴格規則（務必遵守）：
- 只能使用「使用者問題」與「提供的 snippets」中的資訊，不可加入任何外部知識或推測。
- 先判斷每一則結果是否「直接回答/涉及」使用者問題；無關的結果不要硬扯成相關。
- 若相關結果不足以回答問題，請明確說「摘要資訊不足」，並指出缺少什麼資訊。
- 請用繁體中文輸出 3–5 句話。
- 每句話結尾都必須引用來源編號，例如：[1] 或 [1][3]（只引用你判定為相關者）。

使用者問題：
{question}

提供的搜尋結果（只能用以下資訊）：
{snippets_block}
"""

def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "")

def call_llm(prompt: str, temperature: float = 0.0, num_predict: int = 160) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "請用繁體中文回答。"},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()

_cache = {}

def explain_results(question: str, results: list[dict], version: str = "B") -> str:
    top = results[:5]
    key = (
        version.upper(),
        question.strip(),
        tuple((x.get("url", ""), _strip_html(x.get("snippet", ""))) for x in top)
    )
    if key in _cache:
        return _cache[key]

    blocks = []
    for i, r in enumerate(top, start=1):
        title = r.get("title", "")
        url = r.get("url", "")
        snippet = _strip_html(r.get("snippet", ""))
        blocks.append(f"[{i}] Title: {title}\nURL: {url}\nSnippet: {snippet}\n")

    snippets_block = "\n".join(blocks)

    if version.upper() == "A":
        prompt = PROMPT_EXPLAIN_A.format(question=question.strip(), snippets_block=snippets_block)
        ans = call_llm(prompt, temperature=0.9)  # A：故意放鬆＋高溫
    else:
        prompt = PROMPT_EXPLAIN_B.format(question=question.strip(), snippets_block=snippets_block)
        ans = call_llm(prompt, temperature=0.0)  # B：嚴格＋低溫

    _cache[key] = ans
    return ans