import json
import jieba
import pickle
import math
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# 配置
INPUT_FILE = "sports_data.json"
INDEX_FILE = "index.pkl"  # 我們將索引存成二進位檔，讀取速度快

# 停用詞表 (Stopwords)：這些詞太常見，對搜尋沒幫助，要過濾掉 
STOPWORDS = set([
    "的", "了", "和", "是", "就", "都", "而", "及", "與", "著", "或", "一個", "沒有", 
    "我們", "你們", "他們", "它", "在", "有", "也", "這", "那", "為", "之", "但",
    "https", "com", "www", "tw", "news", "html"
])

def clean_text(text):
    """
    前處理：移除標點符號、非中文/英文的雜訊 
    """
    # 只保留中文、英文和數字
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
    return text

def load_data():
    print("正在讀取爬蟲資料...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"成功讀取 {len(data)} 篇新聞")
    return data

def build_index():
    data = load_data()
    
    # 1. 斷詞 (Tokenization) 
    print("正在進行斷詞與前處理...")
    corpus = [] # 存放切好詞的列表，給 TF-IDF 用
    valid_docs = [] # 存放對應的原始文件
    
    for doc in data:
        content = doc['title'] + " " + doc['text'] # 標題和內文一起索引
        content = clean_text(content)
        
        # 使用 jieba 進行中文斷詞
        words = jieba.cut(content)
        
        # 過濾停用詞
        filtered_words = [w for w in words if w not in STOPWORDS and len(w) > 1]
        
        # 重新組合成字串，因為 TfidfVectorizer 吃字串
        corpus.append(" ".join(filtered_words))
        valid_docs.append(doc)

    # 2. 計算 TF-IDF (Compute weights) 
    print("正在計算 TF-IDF 權重...")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # 獲取所有詞彙列表 (Terms)
    terms = vectorizer.get_feature_names_out()
    
    # 3. 建立倒排索引 (Inverted Index) 
    # 結構：{ '關鍵字': [(文件ID, 分數), (文件ID, 分數)...] }
    print("正在建立倒排索引 (Inverted Index)...")
    inverted_index = {}
    
    # tfidf_matrix 是一個稀疏矩陣，我們要把它轉成我們好查的格式
    # 遍歷每一個文件
    rows, cols = tfidf_matrix.nonzero()
    for row, col in zip(rows, cols):
        term = terms[col]
        score = tfidf_matrix[row, col]
        doc_id = row # 這裡用 list 的 index 作為 ID
        
        if term not in inverted_index:
            inverted_index[term] = []
        
        # 儲存 (DocID, Score)
        inverted_index[term].append((doc_id, float(score)))

    # 4. 儲存索引與原始文件
    # 我們需要把 inverted_index 和 valid_docs 都存起來，搜尋時才找得到標題
    save_data = {
        "index": inverted_index,
        "documents": valid_docs
    }
    
    with open(INDEX_FILE, 'wb') as f:
        pickle.dump(save_data, f)
        
    print(f"索引建立完成！共索引了 {len(inverted_index)} 個關鍵字。")
    print(f"檔案已儲存至 {INDEX_FILE}")

if __name__ == "__main__":
    build_index()