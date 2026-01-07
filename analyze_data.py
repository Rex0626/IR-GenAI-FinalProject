import json
from urllib.parse import urlparse
from collections import Counter

def analyze_results():
    # 讀取剛剛爬下來的資料
    with open('sports_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_pages = len(data)
    print(f"=== 爬蟲成果摘要 ===")
    print(f"總頁數: {total_pages}")
    
    # 計算網域分佈
    domains = [urlparse(item['url']).netloc for item in data]
    domain_counts = Counter(domains)
    print("\n網域分佈:")
    for domain, count in domain_counts.items():
        print(f"- {domain}: {count} 頁 ({count/total_pages:.1%})")
        
    # 計算平均字數
    total_chars = sum(len(item['text']) for item in data)
    avg_chars = total_chars / total_pages if total_pages > 0 else 0
    print(f"\n平均每篇字數: {int(avg_chars)} 字")
    print("==================")

if __name__ == "__main__":
    analyze_results()