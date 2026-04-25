import re
import argparse
import requests
import time
import random
import logging
import json
from pathlib import Path
from urllib.parse import urljoin
from bs4 import BeautifulSoup

# --- Sabitler ---
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DATA_DIR       = PROJECT_ROOT / "data"
DEFAULT_OUTPUT = DATA_DIR / "raw_big.txt"

HEADERS = {
    "User-Agent": "turkce-llm-bot/1.0 (contact: test@example.com)",
    "Accept-Language": "tr-TR,tr;q=0.9"
}

logger = logging.getLogger(__name__)
session = requests.Session()
session.headers.update(HEADERS)


# --- Yardımcı fonksiyonlar ---

def normalize_text(text: str) -> str:
    """Basit whitespace normalizasyonu."""
    return re.sub(r'\s+', ' ', text).strip()


def resolve_cli_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def fetch_json(url: str, s: requests.Session | None = None, **kwargs) -> dict | list | None:
    """JSON döndüren bir URL'yi güvenli şekilde çeker."""
    client = s or session
    try:
        res = client.get(url, timeout=10, **kwargs)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        logger.debug("fetch_json hata %s: %s", url, e)
        return None


def flatten_comments(children: list, min_len: int = 30) -> list[str]:
    """Reddit yorum ağacını düz bir metin listesine çevirir."""
    texts: list[str] = []
    for child in children:
        data = child.get("data", {})
        body = data.get("body")
        if body and isinstance(body, str) and len(body) >= min_len:
            texts.append(normalize_text(body))
        # iç içe yorumlar
        replies = data.get("replies")
        if isinstance(replies, dict):
            inner = replies.get("data", {}).get("children", [])
            texts.extend(flatten_comments(inner, min_len))
    return texts


# --- Veri toplama ---

def collect_wikipedia(max_pages: int = 60, depth: int = 1) -> list[str]:
    start_url = "https://tr.wikipedia.org/wiki/Türkiye"
    seen: set[str] = set()
    queue: list[tuple[str, int]] = [(start_url, 0)]
    links: list[str] = []

    while queue and len(links) < max_pages:
        url, d = queue.pop(0)
        if url in seen or d > depth:
            continue
        seen.add(url)
        try:
            res = session.get(url, timeout=10)
            soup = BeautifulSoup(res.text, "html.parser")
        except Exception:
            continue

        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("/wiki/") and ":" not in href:
                full = urljoin("https://tr.wikipedia.org", href)
                if full not in seen:
                    if len(links) < max_pages:
                        links.append(full)
                    queue.append((full, d + 1))

    return list(dict.fromkeys(links))


def scrape_page(url: str) -> str:
    try:
        res = session.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        texts = [p.get_text().strip() for p in soup.find_all("p") if len(p.get_text().strip()) > 50]
        return "\n".join(normalize_text(t) for t in texts)
    except Exception as e:
        logger.debug("scrape_page hata %s: %s", url, e)
        return ""


def get_reddit(subreddits: tuple = ("Turkey",), limit: int = 100, include_comments: bool = True) -> str:
    all_texts: list[str] = []
    s = requests.Session()
    s.headers.update({"User-Agent": "turkce-llm-bot/1.0"})

    for sub in subreddits:
        url = f"https://www.reddit.com/r/{sub}/top/.json?limit={min(limit, 100)}&t=all"
        data = fetch_json(url, s)
        if not data:
            continue

        for child in data.get("data", {}).get("children", []):
            post = child.get("data", {})
            title    = normalize_text(post.get("title", ""))
            selftext = normalize_text(post.get("selftext", ""))
            if title:    all_texts.append(title)
            if selftext: all_texts.append(selftext)

            if include_comments:
                permalink = post.get("permalink")
                if permalink:
                    cjson = fetch_json(f"https://www.reddit.com{permalink}.json?limit=200", s)
                    if isinstance(cjson, list) and len(cjson) > 1:
                        children_data = cjson[1].get("data", {}).get("children", [])
                        all_texts.extend(flatten_comments(children_data))

            time.sleep(random.uniform(0.5, 1.0))
        time.sleep(random.uniform(1.0, 2.0))

    return "\n".join(all_texts)


def main():
    parser = argparse.ArgumentParser(description="Veri toplayıcı: Wikipedia + Reddit")
    parser.add_argument("--wiki-pages",   type=int,   default=60,        help="Wikipedia sayfa limiti")
    parser.add_argument("--wiki-depth",   type=int,   default=1,         help="Wikipedia tarama derinliği")
    parser.add_argument("--subreddits",   nargs="*",  default=["Turkey"],help="Reddit subreddit listesi")
    parser.add_argument("--reddit-limit", type=int,   default=100,       help="Her subreddit için post sayısı")
    parser.add_argument("--output",       default=str(DEFAULT_OUTPUT),    help="Çıktı dosyası")
    args = parser.parse_args()

    all_texts: list[str] = []

    print("Wikipedia çekiliyor...")
    links = collect_wikipedia(max_pages=args.wiki_pages, depth=args.wiki_depth)
    for link in links:
        print("->", link)
        text = scrape_page(link)
        if text:
            all_texts.append(text)
        time.sleep(random.uniform(0.5, 1.2))

    print("Reddit çekiliyor...")
    reddit_text = get_reddit(subreddits=args.subreddits, limit=args.reddit_limit)
    if reddit_text:
        all_texts.append(reddit_text)

    out = resolve_cli_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for t in all_texts:
            f.write(t + "\n")

    print(f"DATASET HAZIR → {out}")


if __name__ == "__main__":
    main()