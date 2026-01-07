import requests

# 這是最簡單的測試，排除所有 Flask 的干擾
try:
    print("正在呼叫 Ollama...")
    response = requests.post(
        "http://127.0.0.1:11434/api/chat",
        json={
            "model": "llama3",  # 記得改成你電腦裡有的模型，如 'llama3', 'mistral' 等
            "messages": [{"role": "user", "content": "你好，你是誰？"}],
            "stream": False
        }
    )
    response.raise_for_status()
    print("測試成功！回應如下：")
    print(response.json()['message']['content'])
except Exception as e:
    print(f"測試失敗，錯誤原因：{e}")