import pickle
import jieba
import time
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple

INDEX_FILE = "index.pkl"
OUTPUT_JSON = "eval_results.json"
OUTPUT_MD = "eval_results.md"

TOP_K = 5  # Precision@5

# 預設 20 個 query（你也可以用 queries.txt 覆蓋）
DEFAULT_QUERIES = [
    "中華隊 張育成",
    "中華隊 經典賽",
    "中職 自由球員",
    "統一獅 蘇智傑 合約",
    "味全龍 補強",
    "富邦悍將 洋投",
    "樂天桃猿 教練",
    "林昱珉 旅美",
    "大谷翔平 轉隊",
    "道奇 大谷翔平",
    "NBA 勇士 交易",
    "湖人 詹姆斯 傷勢",
    "灰熊 莫蘭特 禁賽",
    "TPBL 戰績",
    "SBL 冠軍",
    "羽球 麟洋配",
    "王齊麟 李洋 飯局",
    "戴資穎 傷勢",
    "英超 利物浦 低迷",
    "世界盃 會外賽",
]


def load_index(index_file: str = INDEX_FILE):
    p = Path(index_file)
    if not p.exists():
        raise FileNotFoundError(f"找不到 {index_file}，請先跑 indexer.py 產生索引。")

    with open(index_file, "rb") as f:
        data = pickle.load(f)

    inverted_index = data["index"]
    documents = data["documents"]

    return inverted_index, documents


def tokenize_query(q: str) -> List[str]:
    terms = [t.strip() for t in jieba.lcut(q) if t.strip()]
    return terms


def get_snippet(text: str, terms: List[str], length: int = 140) -> str:
    if not text:
        return ""
    if not terms:
        return (text[:length] + "...") if len(text) > length else text

    low = text.lower()
    start = 0
    for t in terms:
        pos = low.find(t.lower())
        if pos != -1:
            start = max(0, pos - 30)
            break
    snippet = text[start:start + length]
    if start > 0:
        snippet = "..." + snippet
    if start + length < len(text):
        snippet += "..."
    snippet = re.sub(r"\s+", " ", snippet).strip()
    return snippet


def search_topk(query: str, inverted_index: Dict, documents: List[Dict], k: int = TOP_K) -> Tuple[List[Dict], float]:
    """
    用你目前 app 的方式：把每個 query term 的 posting (doc_id, score) 累加，然後排序取 topk
    """
    if not query.strip():
        return [], 0.0

    t0 = time.time()
    terms = tokenize_query(query)
    scores: Dict[int, float] = {}

    for term in terms:
        if term in inverted_index:
            for doc_id, score in inverted_index[term]:
                scores[doc_id] = scores.get(doc_id, 0.0) + float(score)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

    results = []
    for doc_id, score in ranked:
        doc = documents[doc_id]
        results.append({
            "doc_id": doc_id,
            "title": doc.get("title", ""),
            "url": doc.get("url", ""),
            "score": float(score),
            "snippet": get_snippet(doc.get("text", ""), terms),
        })

    return results, time.time() - t0


def read_queries_from_file(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    lines = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def prompt_labels(k: int) -> List[int]:
    """
    讓你輸入 0/1 標注：例如 01101
    也支援：0 1 1 0 1
    """
    while True:
        s = input(f"請輸入 Top{k} 標注（例如 01101；1=相關 0=不相關；或輸入 s 跳過）： ").strip()
        if s.lower() == "s":
            return [-1] * k  # 表示跳過
        if " " in s:
            parts = [x.strip() for x in s.split() if x.strip()]
            if len(parts) != k or any(x not in ("0", "1") for x in parts):
                print("格式不對，請再試一次。")
                continue
            return [int(x) for x in parts]
        else:
            if len(s) != k or any(ch not in "01" for ch in s):
                print("格式不對，請再試一次。")
                continue
            return [int(ch) for ch in s]


def to_markdown_table(rows: List[Dict]) -> str:
    # rows: [{"query":..., "p5":..., "rel_count":..., "labels":...}, ...]
    md = []
    md.append("| # | Query | Relevant in Top5 | Precision@5 |")
    md.append("|---:|---|---:|---:|")
    for i, r in enumerate(rows, 1):
        md.append(f"| {i} | {r['query']} | {r['rel_count']}/5 | {r['p5']:.2f} |")
    return "\n".join(md)


def main():
    inverted_index, documents = load_index(INDEX_FILE)
    print(f"✅ 索引載入完成：{len(documents)} docs, {len(inverted_index)} terms")

    # 如果有 queries.txt 就用它；沒有就用預設 20 個
    file_queries = read_queries_from_file("queries.txt")
    queries = file_queries if file_queries else DEFAULT_QUERIES

    print(f"\n將評估 {len(queries)} 個 queries（Precision@5）")
    print("提示：每題會印 Top5，你只要輸入 0/1 標注即可。\n")

    eval_rows = []
    p5_list = []

    for qi, q in enumerate(queries, 1):
        print("=" * 80)
        print(f"[{qi}/{len(queries)}] Query: {q}")

        results, exec_time = search_topk(q, inverted_index, documents, k=TOP_K)
        print(f"搜尋耗時：{exec_time:.4f} 秒")
        if not results:
            print("⚠️ 沒有結果，這題 Precision@5 記為 0.0")
            eval_rows.append({"query": q, "labels": [0]*TOP_K, "rel_count": 0, "p5": 0.0, "results": []})
            p5_list.append(0.0)
            continue

        for i, r in enumerate(results, 1):
            print(f"\nTop{i}  Score={r['score']:.4f}")
            print(f"Title : {r['title']}")
            print(f"URL   : {r['url']}")
            print(f"Snippet: {r['snippet']}")

        labels = prompt_labels(TOP_K)

        # 跳過就不納入平均（避免你不想標某題）
        if labels[0] == -1:
            print("（已跳過此題，不納入平均）")
            eval_rows.append({"query": q, "labels": labels, "rel_count": 0, "p5": None, "results": results})
            continue

        rel_count = sum(labels)
        p5 = rel_count / TOP_K
        p5_list.append(p5)

        print(f"➡️ Precision@5 = {rel_count}/{TOP_K} = {p5:.2f}")

        eval_rows.append({
            "query": q,
            "labels": labels,
            "rel_count": rel_count,
            "p5": p5,
            "results": results,
        })

    # 平均（只算未跳過的）
    if p5_list:
        avg_p5 = sum(p5_list) / len(p5_list)
    else:
        avg_p5 = 0.0

    print("\n" + "=" * 80)
    print(f"✅ 評估完成：共 {len(p5_list)} 題納入平均")
    print(f"⭐ Average Precision@5 = {avg_p5:.3f}")

    # 輸出 JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "metric": "Precision@5",
            "included_queries": len(p5_list),
            "average_p5": avg_p5,
            "rows": eval_rows,
        }, f, ensure_ascii=False, indent=2)

    # 輸出 Markdown（可貼報告）
    md_rows = []
    for r in eval_rows:
        if r["p5"] is None:
            continue
        md_rows.append({"query": r["query"], "rel_count": r["rel_count"], "p5": r["p5"]})

    md = []
    md.append("## Evaluation: Precision@5\n")
    md.append(to_markdown_table(md_rows))
    md.append(f"\n\n**Average Precision@5 = {avg_p5:.3f}**\n")

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"\n已輸出：{OUTPUT_JSON}、{OUTPUT_MD}")


if __name__ == "__main__":
    main()