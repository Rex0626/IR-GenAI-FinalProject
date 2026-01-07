from flask import Flask, request, render_template_string
from query_rewrite import rewrite_query
from ai_explain import explain_results
import pickle
import jieba
import time
import re
import math

app = Flask(__name__)

# =========================
#   Load index
# =========================
print("正在啟動 Web Server，載入索引中...")
try:
    with open("index.pkl", "rb") as f:
        data = pickle.load(f)
        inverted_index = data["index"]      # term -> list[(doc_id, tfidf_score)]
        documents = data["documents"]       # list of dict: {title,url,text,...}
        total_docs = len(documents)
        total_terms = len(inverted_index)
        print(f"索引載入完成！共 {total_docs} 篇文章，{total_terms} 個關鍵字。")
except FileNotFoundError:
    print("尚未找到 index.pkl，請先執行 indexer.py")
    inverted_index = {}
    documents = []
    total_docs = 0
    total_terms = 0


# =========================
#   Utilities
# =========================
STOP = set([
    "的","有","是","在","和","與","嗎","哪些","什麼","如何","為何","請問",
    "最近","最新","有啥","新聞","消息","快訊","更新","整理","相關","資訊","請"
])

def strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "")

def tokenize_mixed(text: str) -> list[str]:
    """
    中英混合 tokenization：
    - 中文：jieba
    - 英數：regex 抓 NBA/WBC/MLB/2026 這種
    全部英文統一轉小寫，以配合你 indexer 通常會做 lowercasing 的規格。
    """
    # 中文 tokens
    zh = [t.strip() for t in jieba.cut(text) if t.strip()]

    # 英數 tokens
    en = re.findall(r"[A-Za-z]{2,}|\d{2,}", text)
    en = [w.lower() for w in en]

    terms = zh + en
    terms = [t for t in terms if t not in STOP and len(t) >= 2]

    # 去重（保序）
    seen, out = set(), []
    for t in terms:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out

def get_snippet(text, query_terms, length=120):
    """
    產生摘要並標記關鍵字位置（用第一個命中的 term 附近裁切）
    """
    if not query_terms:
        return (text[:length] + "...") if text else ""

    text_low = (text or "").lower()
    start_pos = 0

    for term in query_terms:
        pos = text_low.find(term.lower())
        if pos != -1:
            start_pos = max(0, pos - 20)
            break

    snippet = (text or "")[start_pos : start_pos + length]
    if start_pos > 0:
        snippet = "..." + snippet
    if start_pos + length < len(text or ""):
        snippet = snippet + "..."
    return snippet


# =========================
#   Core: Search (TF-IDF)
# =========================
def search_engine(query, mode="normal", limit=30):
    if not query:
        return [], 0.0

    start_time = time.time()

    # ✅ 用混合斷詞（中文 + 英數）
    query_terms = tokenize_mixed(query)

    scores = {}
    for term in query_terms:
        if term in inverted_index:
            for doc_id, score in inverted_index[term]:
                scores[doc_id] = scores.get(doc_id, 0.0) + float(score)

    # ✅ 沒命中就回空（不要把全體文件塞成 0 分候選）
    if not scores:
        return [], time.time() - start_time

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    WINDOW = 30

    for doc_id, score in ranked:
        doc = documents[doc_id]
        text = doc.get("text", "")

        # Phrase mode (簡易：所有詞都出現且距離近)
        if mode == "phrase" and query_terms:
            ok = True
            positions = []
            text_low = text.lower()

            for term in query_terms:
                pos = text_low.find(term.lower())
                if pos == -1:
                    ok = False
                    break
                positions.append(pos)

            if not ok:
                continue
            if max(positions) - min(positions) > WINDOW:
                continue
            if score <= 0:
                continue

        raw_snippet = get_snippet(text, query_terms)

        highlighted_snippet = raw_snippet
        for term in query_terms:
            if not term:
                continue
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            highlighted_snippet = pattern.sub(
                f'<span class="highlight">{term}</span>',
                highlighted_snippet
            )

        results.append({
            "title": doc.get("title", ""),
            "url": doc.get("url", ""),
            "score": round(float(score), 4),
            "snippet": highlighted_snippet
        })

        if len(results) >= limit:
            break

    return results, time.time() - start_time


# =========================
#   Merge results (multi queries)
# =========================
def merge_results(list_of_result_lists, top_k=30):
    """
    合併多個 query 的結果：同 URL 取最高分，回傳 top_k
    """
    merged = {}
    for results in list_of_result_lists:
        for r in results:
            key = r.get("url", "")
            if not key:
                continue
            if key not in merged or r["score"] > merged[key]["score"]:
                merged[key] = r
    ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]

STOP2 = set(["的","有","是","在","和","與","嗎","哪些","什麼","如何","為何","請問","最近","整理","相關","資訊","有人","有哪些","名單"])

def tokenize_query(q: str) -> list[str]:
    # jieba + 把英文縮寫抓出來（NBA/MLB/WBC 這種）
    toks = [t.strip() for t in jieba.cut(q) if t.strip()]
    toks += re.findall(r"[A-Za-z]{2,}\d*|\d{4}", q)  # NBA / WBC / 2026
    # 去重保序
    out, seen = [], set()
    for t in toks:
        if t not in seen:
            out.append(t); seen.add(t)
    return out

def idf(term: str) -> float:
    # 用倒排索引算 df，再算 idf
    N = max(1, len(documents))
    df = len(inverted_index.get(term, []))
    return math.log((N + 1) / (df + 1)) + 1.0

def pick_must_terms(question: str, top: int = 2) -> list[str]:
    terms = tokenize_query(question)
    terms = [t for t in terms if t not in STOP2 and len(t) >= 2]
    if not terms:
        return []
    # 依 idf 由大到小挑前 top 個（越稀有越「限定」）
    ranked = sorted(terms, key=lambda t: idf(t), reverse=True)
    must = []
    for t in ranked:
        if t not in must:
            must.append(t)
        if len(must) >= top:
            break
    return must

def filter_results_by_must_terms(question: str, results: list[dict], top_k: int = 5) -> list[dict]:
    must = pick_must_terms(question, top=2)
    if not must:
        return results[:top_k]

    def hay(r):
        return (r.get("title","") + " " + strip_html(r.get("snippet",""))).lower()

    # 先嚴格：必須同時包含所有 must terms
    strict = [r for r in results if all(m.lower() in hay(r) for m in must)]
    if len(strict) >= top_k:
        return strict[:top_k]

    # 不夠就放寬：至少包含其中 1 個 must term（但仍比你現在乾淨）
    loose = [r for r in results if any(m.lower() in hay(r) for m in must)]
    if len(loose) >= top_k:
        return loose[:top_k]

    # 再不夠就回傳原本 top_k（避免空）
    return results[:top_k]
# =========================
#   DF-based rerank/filter (No sport list!)
# =========================
def pick_discriminative_terms(question: str, k: int = 2) -> list[str]:
    """
    DF 最小的 k 個詞：越稀有越能區分主題
    df(term) = len(inverted_index[term])
    """
    terms = tokenize_mixed(question)

    scored = []
    for t in terms:
        if t in inverted_index:
            df = len(inverted_index[t])
            scored.append((df, t))
        else:
            scored.append((10**9, t))

    scored.sort(key=lambda x: x[0])  # df 小優先
    return [t for _, t in scored[:k]]

def filter_results_by_keyterms(question: str, results: list[dict], k: int = 2, top_k: int = 5) -> list[dict]:
    """
    用「df 最小」的關鍵詞做重排/過濾：
    - title+snippet 命中越多關鍵詞 → 越前面
    - 不會整批砍光：不足用原順序補滿 top_k
    """
    key_terms = pick_discriminative_terms(question, k=k)
    if not key_terms:
        return results[:top_k]

    kept = []
    rest = []

    for r in results:
        hay = (r.get("title", "") + " " + strip_html(r.get("snippet", ""))).lower()
        hit = sum(1 for t in key_terms if t.lower() in hay)
        if hit >= 1:
            kept.append((hit, r))
        else:
            rest.append(r)

    kept.sort(key=lambda x: x[0], reverse=True)
    kept_only = [r for _, r in kept]

    out = kept_only[:top_k]
    if len(out) < top_k:
        out.extend(rest[: top_k - len(out)])
    return out


# =========================
#   HTML template (原樣保留)
# =========================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>體育搜尋引擎</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f8f9fa; font-family: "Microsoft JhengHei", sans-serif; }
        .hero-section { background: linear-gradient(135deg, #0d6efd, #0dcaf0); color: white; padding: 60px 0; margin-bottom: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .search-card { background: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 20px rgba(0,0,0,0.1); margin-top: -50px; }
        .result-card { border: none; border-left: 5px solid #0d6efd; margin-bottom: 20px; transition: transform 0.2s; }
        .result-card:hover { transform: translateY(-3px); box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
        .result-title { font-size: 1.25rem; font-weight: bold; text-decoration: none; color: #0d6efd; }
        .result-url { font-size: 0.85rem; color: #28a745; margin-bottom: 5px; }
        .result-snippet { color: #555; font-size: 0.95rem; line-height: 1.6; }
        .highlight { color: #d63384; font-weight: bold; background-color: #fff0f5; padding: 0 2px; border-radius: 3px; }
        .stats-badge { font-size: 0.9rem; background: rgba(255,255,255,0.2); padding: 5px 15px; border-radius: 20px; }
        .score-badge { font-size: 0.8rem; background: #6c757d; color: white; padding: 2px 8px; border-radius: 4px; }
    </style>
</head>
<body>

    <div class="hero-section text-center">
        <div class="container">
            <h1 class="display-4 fw-bold">🏀 體育專欄 Search Engine</h1>
            <p class="lead">專屬於體育新聞的垂直搜尋引擎</p>
            <div class="mt-3">
                <span class="stats-badge">📚 已索引文章：{{ total_docs }} 篇</span>
                <span class="stats-badge ms-2">🔑 關鍵字庫：{{ total_terms }} 個</span>
            </div>
        </div>
    </div>

    <div class="container">
        <div class="row justify-content-center">
            <div class="col-md-8">
                <div class="search-card">
                    <form action="/" method="get" class="d-flex flex-wrap gap-2">
                        <input type="text" name="q" class="form-control form-control-lg"
                            value="{{ query }}"
                            placeholder="輸入關鍵字 (例如: 中華隊, 大谷翔平)..." autofocus>

                        <select name="mode" class="form-select form-select-lg" style="max-width: 180px;">
                            <option value="normal" {% if mode != 'phrase' %}selected{% endif %}>一般搜尋</option>
                            <option value="phrase" {% if mode == 'phrase' %}selected{% endif %}>片語搜尋</option>
                        </select>

                        <div class="form-check align-self-center ms-2">
                            <input class="form-check-input" type="checkbox" name="use_ai" value="1" id="use_ai"
                                    {% if use_ai == '1' %}checked{% endif %}>
                            <label class="form-check-label" for="use_ai">使用 GenAI 改寫</label>
                        </div>

                        <select name="pv" class="form-select form-select-lg" style="max-width: 160px;">
                            <option value="A" {% if pv == 'A' %}selected{% endif %}>Prompt A</option>
                            <option value="B" {% if pv != 'A' %}selected{% endif %}>Prompt B</option>
                        </select>

                        <button type="submit" class="btn btn-primary btn-lg px-4">搜尋</button>
                    </form>
                </div>
            </div>
        </div>

        {% if genai_queries and use_ai == '1' %}
        <div class="row justify-content-center mt-3">
            <div class="col-md-8">
                <div class="alert alert-info">
                <b>🤖 GenAI 產生的 Keyword Queries (Prompt {{ pv }})</b>
                <ul class="mb-0">
                    {% for qq in genai_queries %}
                    <li>{{ qq }}</li>
                    {% endfor %}
                </ul>
                </div>
            </div>
        </div>
        {% endif %}

        {% if explanation %}
        <div class="row justify-content-center mt-3">
            <div class="col-md-8">
                <div class="card shadow-sm mb-4">
                    <div class="card-body">
                        <h5 class="fw-bold mb-2">🤖 GenAI 結果解釋（僅根據 snippets）</h5>
                        <div class="text-muted" style="line-height:1.7;">{{ explanation }}</div>
                    </div>
                </div>
            </div>
        </div>
        {% endif %}

        {% if query %}
        <div class="row justify-content-center mt-3">
            <div class="col-md-8">
                <p class="text-muted mb-4">
                    🔍 搜尋 "<b>{{ query }}</b>" 找到 {{ results|length }} 筆結果
                    <small>(耗時 {{ "%.4f"|format(exec_time) }} 秒，模式：{{ '片語搜尋' if mode == 'phrase' else '一般搜尋' }})</small>
                </p>

                {% for res in results %}
                <div class="card result-card shadow-sm">
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-start">
                            <a href="{{ res.url }}" class="result-title" target="_blank">{{ res.title }}</a>
                            <span class="score-badge" title="Relevance Score">Score: {{ res.score }}</span>
                        </div>
                        <div class="result-url text-truncate">{{ res.url }}</div>
                        <div class="result-snippet">{{ res.snippet | safe }}</div>
                    </div>
                </div>
                {% endfor %}

                {% if results|length == 0 %}
                <div class="text-center py-5 text-muted">
                    <h4>🤷‍♂️ 找不到相關新聞</h4>
                    <p>試試看其他關鍵字？例如：NBA, 棒球, 冠軍</p>
                </div>
                {% endif %}
            </div>
        </div>
        {% endif %}

        <footer class="text-center mt-5 mb-4 text-muted small">
            <p>&copy; 2025 Term Project II - Information Retrieval</p>
        </footer>
    </div>

</body>
</html>
"""


# =========================
#   Route
# =========================
@app.route("/")
def home():
    query = request.args.get("q", "")
    mode = request.args.get("mode", "normal")
    use_ai = request.args.get("use_ai", "0")
    pv = request.args.get("pv", "B")

    results = []
    exec_time = 0.0
    genai_queries = []
    explanation = ""

    if query:
        t0 = time.time()

        if use_ai == "1":
            rewritten = rewrite_query(query, version=pv)
            genai_queries = rewritten.get("queries", []) or [query]

            all_results = []
            for q2 in genai_queries:
                r, _ = search_engine(q2, mode=mode, limit=30)
                all_results.append(r)

            merged = merge_results(all_results, top_k=30)     # 先拿多一點
            results = filter_results_by_must_terms(query, merged, top_k=5)
        else:
            genai_queries = [query]
            r, _ = search_engine(query, mode=mode, limit=30)
            results = filter_results_by_keyterms(query, r, k=2, top_k=5)

        exec_time = time.time() - t0

        if results:
            explanation = explain_results(query, results, version=pv)

        print("DF key terms:", pick_discriminative_terms(query, k=2), "Results:", len(results))

    return render_template_string(
        HTML_TEMPLATE,
        query=query,
        results=results,
        exec_time=exec_time,
        total_docs=total_docs,
        total_terms=total_terms,
        mode=mode,
        use_ai=use_ai,
        pv=pv,
        genai_queries=genai_queries,
        explanation=explanation,
    )


if __name__ == "__main__":
    app.run(debug=False, port=5000)