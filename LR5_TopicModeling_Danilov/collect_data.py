"""
Сбор корпусов для тематического моделирования.

Дособирает данные ЛР4 до объёма, требуемого заданием:
  * news     - статьи Habr (заголовок, лид, хабы, теги), 200+ документов;
  * products - товары Wildberries (название, бренд, описание) + отзывы, 200+ документов.

Запуск:
    python collect_data.py --dataset news --max-items 220
    python collect_data.py --dataset products --max-items 220 --reviews 30
    python collect_data.py --dataset all
"""

import argparse
import csv
import html
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import requests

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "ru-RU,ru;q=0.9",
}

PRODUCT_QUERIES = [
    "ноутбук", "смартфон", "наушники беспроводные", "робот пылесос",
    "кофемашина", "кроссовки мужские", "рюкзак городской", "электрический чайник",
    "мультиварка", "монитор", "детская коляска", "крем для лица",
    "палатка туристическая", "настольная игра", "корм для кошек",
]

TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


def clean_html(text):
    text = html.unescape(TAG_RE.sub(" ", text or ""))
    return SPACE_RE.sub(" ", text).strip()


def get_json(session, url, params=None, timeout=15, retries=3):
    for attempt in range(retries):
        try:
            response = session.get(url, params=params, timeout=timeout)
            if response.status_code == 429:
                time.sleep(2 + attempt * 3)
                continue
            if response.status_code != 200:
                return None
            return response.json()
        except (requests.RequestException, ValueError):
            time.sleep(1 + attempt)
    return None


# --------------------------------------------------------------------------- Habr

def habr_api_page(session, page):
    data = get_json(
        session,
        "https://habr.com/kek/v2/articles/",
        params={"sort": "all", "fl": "ru", "hl": "ru", "page": page, "perPage": 20},
    )
    if not data:
        return []

    ids = data.get("publicationIds") or data.get("articleIds") or []
    refs = data.get("publicationRefs") or data.get("articleRefs") or {}
    items = []
    for article_id in ids:
        ref = refs.get(str(article_id)) or refs.get(article_id)
        if not ref:
            continue
        lead = (ref.get("leadData") or {}).get("textHtml", "")
        items.append({
            "title": clean_html(ref.get("titleHtml", "")),
            "summary": clean_html(lead),
            "hubs": [h.get("title", "") for h in ref.get("hubs") or []],
            "tags": [clean_html(t.get("titleHtml", "")) for t in ref.get("tags") or []],
            "date": ref.get("timePublished", ""),
            "url": f"https://habr.com/ru/articles/{article_id}/",
            "source": "Habr",
        })
    return items


def habr_html_page(session, page):
    try:
        from parsel import Selector
    except ImportError:
        return []

    url = "https://habr.com/ru/articles/" if page == 1 else f"https://habr.com/ru/articles/page{page}/"
    try:
        response = session.get(url, timeout=15)
    except requests.RequestException:
        return []
    if response.status_code != 200:
        return []

    selector = Selector(text=response.text)
    items = []
    for article in selector.css("article.tm-articles-list__item"):
        title = article.xpath("string(.//a[contains(@class, 'tm-title__link')])").get("").strip()
        link = article.css("a.tm-title__link::attr(href)").get("")
        if not title or not link:
            continue
        items.append({
            "title": title,
            "summary": clean_html(" ".join(article.css(".tm-article-body *::text, .article-formatted-body *::text").getall())),
            "hubs": [x.strip() for x in article.css(".tm-publication-hub__link span::text").getall() if x.strip()],
            "tags": [],
            "date": article.css("time::attr(datetime)").get(""),
            "url": urljoin(response.url, link),
            "source": "Habr",
        })
    return items


def habr_article_text(session, url, limit=3000):
    try:
        from parsel import Selector
    except ImportError:
        return ""
    try:
        response = session.get(url, timeout=15)
    except requests.RequestException:
        return ""
    if response.status_code != 200:
        return ""
    selector = Selector(text=response.text)
    parts = selector.css("div.article-formatted-body *::text, #post-content-body *::text").getall()
    return SPACE_RE.sub(" ", " ".join(parts)).strip()[:limit]


def collect_news(max_items, delay):
    session = requests.Session()
    session.headers.update(HEADERS)

    collected, seen = [], set()
    page = 1
    empty_pages = 0
    while len(collected) < max_items and page <= 60 and empty_pages < 3:
        items = habr_api_page(session, page)
        source = "api"
        if not items:
            items = habr_html_page(session, page)
            source = "html"
        added = 0
        for item in items:
            if item["url"] in seen or not item["title"]:
                continue
            seen.add(item["url"])
            if len(item["summary"]) < 200:
                item["text"] = habr_article_text(session, item["url"])
                time.sleep(delay)
            collected.append(item)
            added += 1
        empty_pages = empty_pages + 1 if added == 0 else 0
        print(f"[news] страница {page} ({source}): +{added}, всего {len(collected)}")
        page += 1
        time.sleep(delay)

    return collected[:max_items]


# --------------------------------------------------------------------- Wildberries
#
# Поиск WB закрыт антибот-защитой (wbaas) и без браузерной сессии отвечает 403,
# поэтому ID товаров берутся пауком EcommerceSpider из ЛР4 (Playwright открывает
# обычные страницы поиска). Карточки товаров (card.json) и отзывы отдаются
# публичными CDN/API и загружаются обычными HTTP-запросами.

LR4_DIR = BASE_DIR.parent / "LR4_Scrapy_TwoSpiders_Danilov"
WB_SEARCH_DIR = DATA_DIR / "wb_search"
WB_CARD_HOSTS = [f"https://mow-basket-cdn-{i:02d}.geobasket.ru" for i in (1, 2, 8, 15, 28, 33)]
WB_ID_RE = re.compile(r"wildberries\.ru/catalog/(\d+)/")


def ids_from_csv(path):
    ids = []
    try:
        with open(path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f, delimiter=";"):
                match = WB_ID_RE.search(row.get("url", ""))
                if match:
                    ids.append(int(match.group(1)))
    except OSError:
        pass
    return ids


def run_lr4_spider(query, index, per_query):
    WB_SEARCH_DIR.mkdir(parents=True, exist_ok=True)
    output = WB_SEARCH_DIR / f"{index:02d}.csv"
    if output.exists() and ids_from_csv(output):
        return output

    print(f"[products] паук ЛР4 ищет '{query}' на Wildberries...")
    command = [
        sys.executable, "run_spider.py", "--spider", "ecommerce",
        "--category", query, "--stores", "wildberries",
        "--max-items", str(per_query), "--output", str(output),
    ]
    try:
        subprocess.run(command, cwd=LR4_DIR, timeout=600, check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"[products] '{query}': паук ЛР4 не запустился ({error})")
    return output


def collect_wb_ids(per_query, extra_csv):
    groups = []
    if not (LR4_DIR / "run_spider.py").exists():
        print(f"[products] не найден паук ЛР4: {LR4_DIR}")
    else:
        for index, query in enumerate(PRODUCT_QUERIES, start=1):
            ids = ids_from_csv(run_lr4_spider(query, index, per_query))
            print(f"[products] поиск '{query}': {len(ids)} ID товаров")
            groups.append((query, ids))

    for path in extra_csv:
        ids = ids_from_csv(path)
        print(f"[products] {path}: {len(ids)} ID товаров")
        groups.append(("ноутбук", ids))
    return groups


def wb_card(session, nm_id):
    vol, part = nm_id // 100000, nm_id // 1000
    path = f"/vol{vol}/part{part}/{nm_id}/info/ru/card.json"
    for host in WB_CARD_HOSTS:
        data = get_json(session, host + path, timeout=10, retries=1)
        if data:
            return data
    return None


_feedback_hosts = {}


def wb_feedbacks(session, imt_id, limit):
    if imt_id not in _feedback_hosts:
        hosts = get_json(session, "https://feedback-bt.wildberries.ru/feedback/api/v2/host",
                         params={"imt": imt_id}, retries=1)
        _feedback_hosts[imt_id] = hosts if isinstance(hosts, list) and hosts else ["https://feedback-view-01.wb.ru"]

    for host in _feedback_hosts[imt_id]:
        data = get_json(session, f"{host}/feedbacks/v2/{imt_id}", retries=2)
        if not data or not data.get("feedbacks"):
            continue
        texts = []
        for fb in data["feedbacks"]:
            parts = [fb.get("text", ""), fb.get("pros", ""), fb.get("cons", "")]
            text = SPACE_RE.sub(" ", " ".join(p for p in parts if p)).strip()
            if len(text) >= 20:
                texts.append({"text": text, "rating": fb.get("productValuation")})
            if len(texts) >= limit:
                break
        return texts, data.get("valuation")
    return [], None


def collect_products(max_items, reviews_limit, delay, extra_csv):
    session = requests.Session()
    session.headers.update(HEADERS)

    per_query = max(5, max_items // len(PRODUCT_QUERIES) + 6)
    groups = collect_wb_ids(per_query, extra_csv)

    target_per_group = max(1, max_items // max(1, len(groups)) + 1)
    collected, seen_nm, seen_imt = [], set(), set()

    for category, ids in groups:
        taken = 0
        for nm_id in ids:
            if taken >= target_per_group or len(collected) >= max_items:
                break
            if nm_id in seen_nm:
                continue
            seen_nm.add(nm_id)

            card = wb_card(session, nm_id)
            if not card or card.get("imt_id") in seen_imt:
                continue
            reviews, rating = wb_feedbacks(session, card["imt_id"], reviews_limit)
            if len(reviews) < 3:
                continue

            seen_imt.add(card["imt_id"])
            collected.append({
                "title": card.get("imt_name", ""),
                "brand": (card.get("selling") or {}).get("brand_name", ""),
                "category": card.get("subj_name") or category,
                "description": clean_html(card.get("description", "")),
                "rating": rating,
                "reviews": reviews,
                "url": f"https://www.wildberries.ru/catalog/{nm_id}/detail.aspx",
                "source": "Wildberries",
            })
            taken += 1
            time.sleep(delay)

        print(f"[products] '{category}': +{taken} товаров с отзывами, всего {len(collected)}")

    return collected


def save(items, name):
    if not items:
        print(f"Ничего не собрано для '{name}', файл не перезаписан")
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{name}.json"
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Сохранено {len(items)} записей -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Сбор корпусов для ЛР5")
    parser.add_argument("--dataset", choices=["news", "products", "all"], default="all")
    parser.add_argument("--max-items", type=int, default=220)
    parser.add_argument("--reviews", type=int, default=30, help="отзывов на товар")
    parser.add_argument("--delay", type=float, default=0.7)
    parser.add_argument("--ids-csv", nargs="*", default=[str(BASE_DIR.parent / "LR4_Scrapy_TwoSpiders_Danilov" / "output" / "marketplaces.csv")],
                        help="CSV паука ЛР4 с URL товаров WB (дополнительные ID)")
    args = parser.parse_args()

    if args.dataset in ("news", "all"):
        news = collect_news(args.max_items, args.delay)
        if len(news) < 200:
            print(f"ВНИМАНИЕ: собрано только {len(news)} новостей", file=sys.stderr)
        save(news, "news")

    if args.dataset in ("products", "all"):
        products = collect_products(args.max_items, args.reviews, args.delay, args.ids_csv)
        if len(products) < 200:
            print(f"ВНИМАНИЕ: собрано только {len(products)} товаров", file=sys.stderr)
        save(products, "products")


if __name__ == "__main__":
    main()
