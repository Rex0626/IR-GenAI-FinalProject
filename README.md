# 🏀 體育新聞搜尋引擎 (Sports News Search Engine)

這是 **資訊檢索與生成式 AI (IR & GenAI)** 課程的期末專案 II。
本專案實作了一個完整的搜尋引擎，包含網路爬蟲、倒排索引建置以及 TF-IDF 排序演算法。

## 📂 專案架構
* `crawler.py`: **Web Spider** - 負責爬取 Yahoo 運動、ETtoday 等體育新聞網頁，並過濾非文章頁面。
* `analyze_data.py`: **Data Analysis** - 分析 `sports_data.json`，計算總頁數、網域分佈與平均字數，用於產出報告數據 。
* `indexer.py`: **Indexer** - 使用 Jieba 斷詞並計算 TF-IDF，建立倒排索引 (Inverted Index)。
* `search.py`: **Search Logic** - 核心搜尋演算法與排名邏輯 (Cosine Similarity)。
* `app.py`: **Web Interface** - 基於 Flask 的網頁搜尋介面，支援關鍵字高亮與摘要顯示。
* `sports_data.json`: 爬蟲抓取的 1,000 篇新聞資料集。

## 🚀 如何執行
1. **如果是第一次執行，請先安裝套件**
```bash
pip install -r requirements.txt
```

2. **爬蟲收集相關資料(如果沒有 sports_data.json)**
```bash
python crawler.py
```

3. **建立索引 (如果沒有 index.pkl)**
```bash
python indexer.py
```

## 安裝 Ollama（本機 GenAI）

Ollama 用來在本機跑 LLM（本專題用它提供 HTTP API：`http://127.0.0.1:11434`）。 [oai_citation:1‡docs.ollama.com](https://docs.ollama.com/?utm_source=chatgpt.com)

### macOS
1. 到 Ollama 官方下載頁下載 macOS 版並安裝（拖到 Applications）。 [oai_citation:2‡GitHub](https://github.com/ollama/ollama?tab=readme-ov-file&utm_source=chatgpt.com)
2. 安裝後打開 Ollama（會在選單列看到圖示）。

### Windows
1. 到 Ollama 官方下載頁下載 Windows 安裝檔並安裝。 [oai_citation:3‡GitHub](https://github.com/ollama/ollama?tab=readme-ov-file&utm_source=chatgpt.com)
2. 安裝後啟動 Ollama（會在系統列看到圖示）。


### Ollama 的部分：
1. 首先確認 Ollama 是否有模型
```ollama list```

2. 如果沒有模型，下載（例：gemma2 2b）
```ollama pull gemma2:2b```

3. 開啟 Ollama 服務（Terminal 1）
```ollama serve```
這個視窗要一直開著不要關。

**注意**:如果執行ollama serve的時候發現到port被占用。請先去右下角工具列，對 Ollama 圖示（一隻羊駝）按右鍵選擇 Quit Ollama。

**啟動搜尋引擎 (注意：如果沒有index.pkl，會跑不出結果)**
（Terminal 2）
先執行app.py，再打開瀏覽器訪問 http://127.0.0.1:5000/ 即可開始搜尋。
```python app.py```

- Terminal 1 跑 `ollama serve`（LLM 服務）  
- Terminal 2 跑 `python app.py`（Flask 網頁）  

5. **數據分析 (Optional)**
若想查看爬蟲結果的統計數據（如網域分佈、總頁數），可執行：
```Bash
python analyze_data.py
產生 Prompt A/B 比較結果（Prompt Engineering Analysis）
terminal(2)
python -u pe_analysis.py
```
