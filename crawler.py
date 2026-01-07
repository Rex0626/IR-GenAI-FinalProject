import json
import time
import re
import datetime
import requests
from collections import deque
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from urllib import robotparser

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError


# =========================
# 設定
# =========================
SEED_URLS = [
    "https://tw.sports.yahoo.com/",
    "https://sports.ltn.com.tw/",
    "https://tsna.com/",
    "https://today.line.me/tw/v3/tab/sports",
    "https://udn.com/news/cate/2/7227",
    "https://sports.ettoday.net",
    "https://www.hk01.com/channel/20/即時體育",
    "https://www.sportsv.net",
]

MAX_ARTICLES = 1000
MAX_VISITS = 25000                    # 防止一直繞圈
MIN_DELAY_PER_DOMAIN = 2.2            # 符合作業 >=2s
OUTPUT_FILE = "sports_data.json"
USER_AGENT = "Mozilla/5.0 (Education Purpose; IR Project)"

HEADLESS = True                       # 若被擋可改 False
NAV_TIMEOUT_MS = 15000

# round-robin + queue 控制
MAX_QUEUE_PER_DOMAIN = 4000           # 每個網站最多保留多少待爬 URL（避免爆記憶體）
MAX_TEXT_LEN = 2000                   # 主文最大長度（砍掉推薦通常在尾巴）
MIN_TEXT_LEN = 200                    # 太短視為抓不到主文


# =========================
# 針對「非體育」的核心修正參數
# =========================
# UDN 只允許這些 story 類別 id（避免地震/社會/政治）
# 你可以依實際抓到的運動 story 類別再加（保守做法：先少）
UDN_SPORT_STORY_IDS = {
    "7005",  # 常見運動/體育 story
    "7006",  # 另一個常見運動 story
}

# Yahoo 常見非文章頁（球員/球隊/賽程/數據/比分...）
YAHOO_DENY_PATH_PATTERNS = [
    re.compile(r"/soccer/players/"),
    re.compile(r"/(players|teams)/"),
    re.compile(r"/(standings|schedule|stats|scores)/"),
]

# hk01 只允許體育頻道 channel/20/ 的「列表頁」；其他 channel 一律擋
HK01_DENY_CHANNEL = re.compile(r"^/channel/(?!20/)\d+/")

# （可選）如果你想更嚴格：hk01 文章頁路徑若明顯是電影/娛樂也擋
HK01_DENY_KEYWORDS_IN_PATH = [
    "電影", "娛樂", "藝文", "生活", "旅遊", "美食",
]


# =========================
# URL / robots 工具
# =========================
def canonicalize_url(url: str) -> str:
    """移除 fragment、常見追蹤參數，降低重複（含 bcmt）"""
    try:
        p = urlparse(url)
        if p.scheme not in ("http", "https"):
            return ""

        drop_keys = {
            "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
            "fbclid", "gclid", "guccounter", "guce_referrer", "guce_referrer_sig",
            "spm", "from", "ref", "src", "feature",
            "bcmt",  # Yahoo 常見：?bcmt=1 造成同文重複
        }

        q = []
        for k, v in parse_qsl(p.query, keep_blank_values=True):
            if k.lower() not in drop_keys:
                q.append((k, v))
        query = urlencode(sorted(q))

        clean = urlunparse((p.scheme, p.netloc, p.path, "", query, ""))
        return clean.rstrip("/")
    except Exception:
        return ""


def get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def should_skip_url(url: str) -> bool:
    """快速排除明顯不是文章或不想爬的路徑（含 domain-specific 過濾）"""
    try:
        p = urlparse(url)
        dom = p.netloc.lower()
        path = (p.path or "")
        u = url.lower()

        # 1) 非 http(s) 擋掉
        if p.scheme not in ("http", "https"):
            return True

        # 2) 擋資源檔
        if re.search(r"\.(jpg|jpeg|png|gif|webp|svg|mp4|mov|avi|m3u8|mp3|pdf)(\?|$)", path.lower()):
            return True

        # 3) 通用黑名單（注意：不要放 line.me，會誤傷 today.line.me）
        bad = [
            "/video", "/tv/", "/live", "/livestream",
            "login", "member", "signup", "register",
            "shop", "job", "privacy", "terms",
            "facebook.com", "accounts.google",
        ]
        if any(b in u for b in bad):
            return True

        # 4) 只擋 line.me（短鏈域名）；today.line.me 允許
        if dom == "line.me":
            return True

        # 5) Yahoo：擋球員/球隊/戰績/賽程/數據等頁
        if dom in ("tw.sports.yahoo.com", "sports.yahoo.com"):
            pl = path.lower()
            if any(pat.search(pl) for pat in YAHOO_DENY_PATH_PATTERNS):
                return True

        # 6) UDN：只允許體育 story 類別 id
        if dom == "udn.com":
            m = re.match(r"^/news/story/(\d+)/\d+", path)
            if m and m.group(1) not in UDN_SPORT_STORY_IDS:
                return True

        # 7) hk01：只允許體育頻道列表頁（channel/20/），其他頻道列表頁擋
        if dom == "www.hk01.com":
            if HK01_DENY_CHANNEL.search(path):
                return True
            # 可選更嚴格：如果路徑包含明顯非體育關鍵字就擋
            # （避免你 frontier 撿到電影/娛樂文章）
            for kw in HK01_DENY_KEYWORDS_IN_PATH:
                if kw in path:
                    return True

        return False
    except Exception:
        return True


class RobotsCache:
    def __init__(self, user_agent: str):
        self.user_agent = user_agent
        self.cache = {}

    def can_fetch(self, url: str) -> bool:
        dom = get_domain(url)
        if not dom:
            return False

        if dom not in self.cache:
            rp = robotparser.RobotFileParser()
            robots_url = f"https://{dom}/robots.txt"
            try:
                r = requests.get(robots_url, headers={"User-Agent": self.user_agent}, timeout=6)
                rp.parse(r.text.splitlines())
            except Exception:
                rp = None
            self.cache[dom] = rp

        rp = self.cache[dom]
        if rp is None:
            return True
        return rp.can_fetch(self.user_agent, url)


# =========================
# 文章 URL 規則（站點特化）
# =========================
ARTICLE_PATTERNS = {
    "tw.sports.yahoo.com": [
        re.compile(r"^https?://tw\.sports\.yahoo\.com/news/.+"),
        re.compile(r"^https?://tw\.sports\.yahoo\.com/.+-\d{6,}(\.html)?$"),
    ],
    "sports.yahoo.com": [
        re.compile(r"^https?://sports\.yahoo\.com/news/.+"),
        re.compile(r"^https?://sports\.yahoo\.com/.+-\d{6,}(\.html)?$"),
    ],
    "sports.ltn.com.tw": [
        re.compile(r"^https?://sports\.ltn\.com\.tw/news/.+"),
    ],
    "tsna.com": [
        re.compile(r"^https?://tsna\.com/article/\d+"),
        re.compile(r"^https?://tsna\.com/article/\d+/.+"),
    ],
    "today.line.me": [
        re.compile(r"^https?://today\.line\.me/tw/v2/article/.+"),
        re.compile(r"^https?://today\.line\.me/tw/v3/article/.+"),
    ],
    "udn.com": [
        re.compile(r"^https?://udn\.com/news/story/\d+/\d+"),
    ],
    "sports.ettoday.net": [
        re.compile(r"^https?://sports\.ettoday\.net/news/\d+"),
    ],
    "www.hk01.com": [
        re.compile(r"^https?://www\.hk01\.com/article/\d+"),
        re.compile(r"^https?://www\.hk01\.com/.+/article/\d+"),
    ],
    "www.sportsv.net": [
        re.compile(r"^https?://www\.sportsv\.net/articles/\d+"),
        re.compile(r"^https?://www\.sportsv\.net/articles/\d+/.+"),
    ],
}


def is_article_url(url: str) -> bool:
    # ✅ 保險：先做 should_skip_url，避免 Yahoo players、UDN 非體育 story、hk01 非體育頻道漏網
    if should_skip_url(url):
        return False

    dom = get_domain(url)
    pats = ARTICLE_PATTERNS.get(dom)
    if not pats:
        return False
    return any(p.match(url) for p in pats)


# =========================
# 主文抽取（selector + 去噪 + 砍推薦）
# =========================
SITE_CONTENT_SELECTORS = {
    "tw.sports.yahoo.com": [".caas-body", "article"],
    "sports.yahoo.com": [".caas-body", "article"],
    "sports.ltn.com.tw": [".news_content", ".text", "article"],
    "tsna.com": ["article", ".article", ".article-content"],
    "today.line.me": ["article", "main article", ".article-body", "[data-testid='articleBody']"],
    "udn.com": ["#story_body_content", "section.article-content__editor", "div.article-content__editor", "article"],
    "sports.ettoday.net": ["#news-content", ".story", "article"],
    "www.hk01.com": ["article", ".article-content", ".article-detail"],
    "www.sportsv.net": ["article", ".article-content", ".post-content", ".content"],
}

CUT_KEYWORDS = [
    "猜你喜歡", "延伸閱讀", "相關文章", "相關新聞",
    "更多新聞", "熱門新聞", "你可能會喜歡", "相關報導",
    "發起對話", "相關閱讀", "查看原始文章",
]


def post_clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    for kw in CUT_KEYWORDS:
        idx = text.find(kw)
        if idx != -1:
            text = text[:idx].strip()
            break

    if len(text) > MAX_TEXT_LEN:
        text = text[:MAX_TEXT_LEN].rstrip()

    return text


def auto_scroll(page, times=2):
    for _ in range(times):
        page.mouse.wheel(0, 1800)
        page.wait_for_timeout(800)


def extract_title(page) -> str:
    try:
        h1 = page.locator("h1").first
        t = h1.inner_text(timeout=1500).strip()
        if t:
            return t
    except Exception:
        pass
    try:
        return page.title().strip() or "No Title"
    except Exception:
        return "No Title"


def extract_article_text(page, url: str) -> str:
    dom = get_domain(url)
    selectors = SITE_CONTENT_SELECTORS.get(dom, ["article", "main", "body"])

    # 刪除噪音區塊（廣告/側欄/推薦/導覽等）
    noise_selectors = [
        "script", "style", "nav", "header", "footer", "aside",
        "[role='navigation']", "[aria-label*='navigation']",
        ".related", ".recommend", ".recommendation", ".sidebar",
        ".share", ".social", ".ad", ".ads", ".advertisement",
    ]
    for sel in noise_selectors:
        try:
            page.eval_on_selector_all(sel, "els => els.forEach(e => e.remove())")
        except Exception:
            pass

    best = ""
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            txt = loc.inner_text(timeout=2500).strip()
            if len(txt) > len(best):
                best = txt
            if len(best) >= 400:
                break
        except Exception:
            continue

    if not best:
        try:
            best = page.locator("body").inner_text(timeout=2500).strip()
        except Exception:
            best = ""

    return post_clean_text(best)


# =========================
# Round-robin 主爬蟲
# =========================
def crawl():
    # 允許的網域（以 seed 的 domain 為主）
    allow_domains = {get_domain(u) for u in SEED_URLS if get_domain(u)}
    # Yahoo 可能跳轉到 sports.yahoo.com
    allow_domains |= {"sports.yahoo.com"}

    robots = RobotsCache(USER_AGENT)
    last_fetch = {}  # domain -> monotonic time

    visited = set()
    saved = 0
    visits = 0
    data = []

    # 每個 domain 一個 queue（Round-robin）
    domain_queues = {d: deque() for d in allow_domains}
    active_domains = deque()          # 輪詢用
    active_set = set()               # 避免重複加入 active_domains

    # 初始化：把 seed 丟到對應 domain queue，並加入輪詢
    for s in SEED_URLS:
        u = canonicalize_url(s)
        if not u:
            continue
        if should_skip_url(u):
            continue
        d = get_domain(u)
        if d in allow_domains:
            domain_queues[d].append(u)
            if d not in active_set:
                active_domains.append(d)
                active_set.add(d)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        context = browser.new_context(
            user_agent=USER_AGENT,
            ignore_https_errors=True,
        )
        page = context.new_page()
        page.set_default_navigation_timeout(NAV_TIMEOUT_MS)

        print(f"開始爬取（Playwright Round-robin），目標：{MAX_ARTICLES} 篇文章…")

        while active_domains and saved < MAX_ARTICLES and visits < MAX_VISITS:
            dom = active_domains.popleft()
            active_set.discard(dom)

            q = domain_queues.get(dom)
            if not q:
                continue
            if not q:
                continue

            url = canonicalize_url(q.popleft())
            if not url:
                if q and dom not in active_set:
                    active_domains.append(dom)
                    active_set.add(dom)
                continue

            # 這個 domain 若還有 URL，先放回輪詢（保證公平）
            if q and dom not in active_set:
                active_domains.append(dom)
                active_set.add(dom)

            if url in visited:
                continue

            if should_skip_url(url):
                visited.add(url)
                continue

            if get_domain(url) != dom:
                continue

            # robots.txt
            if not robots.can_fetch(url):
                visited.add(url)
                continue

            # per-domain delay
            now = time.monotonic()
            prev = last_fetch.get(dom, 0)
            wait = MIN_DELAY_PER_DOMAIN - (now - prev)
            if wait > 0:
                time.sleep(wait)

            print(f"\n[VISIT] ({dom}) {url}")

            try:
                visited.add(url)
                last_fetch[dom] = time.monotonic()
                visits += 1

                page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)

                # 被擋檢測
                try:
                    html = page.content()
                except Exception:
                    html = ""

                if html and ("captcha" in html.lower() or "驗證" in html):
                    print("[BLOCKED] captcha/verification")
                    continue

                # 列表頁稍微滾動增加連結
                if not is_article_url(url):
                    auto_scroll(page, times=2)

                # 擴展 frontier：抓連結，分流到對應 domain queue
                try:
                    hrefs = page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")
                except Exception:
                    hrefs = []

                added_article = 0
                added_other = 0

                for h in hrefs:
                    h = canonicalize_url(h)
                    if not h:
                        continue
                    if should_skip_url(h):
                        continue
                    hd = get_domain(h)
                    if hd not in allow_domains:
                        continue
                    if h in visited:
                        continue

                    dq = domain_queues[hd]
                    if len(dq) >= MAX_QUEUE_PER_DOMAIN:
                        continue

                    # 文章優先塞進該 domain queue 的前面（但仍由 round-robin 控制公平）
                    if is_article_url(h):
                        dq.appendleft(h)
                        added_article += 1
                    else:
                        dq.append(h)
                        added_other += 1

                    if hd not in active_set and len(dq) > 0:
                        active_domains.append(hd)
                        active_set.add(hd)

                print(
                    f"[STATE] saved={saved} visits={visits} active_domains={len(active_domains)} "
                    f"({dom}:q={len(domain_queues[dom])}) added_article={added_article} added_other={added_other}"
                )

                # 只存文章
                if not is_article_url(url):
                    continue

                title = extract_title(page)
                text = extract_article_text(page, url)

                if len(text) < MIN_TEXT_LEN:
                    print("[SKIP] text too short")
                    continue

                data.append({
                    "url": url,
                    "title": title,
                    "text": text,
                    "fetch_time": datetime.datetime.now().isoformat(),
                })
                saved += 1
                print(f"[SAVE] {saved}/{MAX_ARTICLES}  {title[:60]}")

            except PWTimeoutError:
                print("[TIMEOUT] goto timeout")
                continue
            except Exception as e:
                print(f"[ERROR] {type(e).__name__}: {e}")
                continue

        browser.close()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n完成！共存入 {len(data)} 篇文章，輸出：{OUTPUT_FILE}")


if __name__ == "__main__":
    crawl()