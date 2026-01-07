import pickle
import jieba
import time

# 設定
INDEX_FILE = "index.pkl"

def load_index():
    print("正在載入搜尋引擎索引...")
    start_time = time.time()
    with open(INDEX_FILE, 'rb') as f:
        data = pickle.load(f)
    print(f"載入完成！耗時 {time.time() - start_time:.2f} 秒")
    return data['index'], data['documents']

def get_snippet(text, query_terms, length=100):
    """
    產生摘要：嘗試找出關鍵字出現的位置，截取那附近的文字
    """
    # 嘗試找到第一個出現的關鍵字位置
    start_pos = 0
    for term in query_terms:
        pos = text.find(term)
        if pos != -1:
            start_pos = max(0, pos - 10) # 往前多取 10 個字讓語意完整
            break
            
    snippet = text[start_pos : start_pos + length]
    return snippet + "..."

def search(query, inverted_index, documents):
    # 1. 對查詢詞進行斷詞
    query_terms = list(jieba.cut(query))
    print(f"搜尋關鍵字: {query_terms}")
    
    # 2. 計算分數 (累加 TF-IDF)
    # scores 是一個字典：{ doc_id: total_score }
    scores = {}
    
    for term in query_terms:
        if term in inverted_index:
            # 取出這個詞在哪些文章出現過，以及分數
            posting_list = inverted_index[term]
            for doc_id, score in posting_list:
                if doc_id not in scores:
                    scores[doc_id] = 0
                scores[doc_id] += score
    
    # 3. 排序 (Ranking)
    # 依照分數由高到低排序，並只取前 10 名
    ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # 4. 顯示結果
    print(f"\n找到 {len(scores)} 篇相關新聞，顯示前 {len(ranked_results)} 筆：\n" + "="*50)
    
    for rank, (doc_id, score) in enumerate(ranked_results, 1):
        doc = documents[doc_id]
        snippet = get_snippet(doc['text'], query_terms)
        
        print(f"Rank {rank} (Score: {score:.4f})")
        print(f"標題: {doc['title']}")
        print(f"連結: {doc['url']}")
        print(f"摘要: {snippet}")
        print("-" * 50)

if __name__ == "__main__":
    # 程式啟動時先載入索引
    idx, docs = load_index()
    
    while True:
        query = input("\n請輸入搜尋關鍵字 (輸入 q 離開): ").strip()
        if query.lower() == 'q':
            break
        if not query:
            continue
            
        start_search = time.time()
        search(query, idx, docs)
        print(f"搜尋耗時: {time.time() - start_search:.4f} 秒")