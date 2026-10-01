import argparse
import csv
import json
from pathlib import Path
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

SOURCE_NAMES = {
    "wildberries": "Wildberries",
    "ozon": "Ozon",
    "yandexmarket": "Yandex Market",
    "yandex": "Yandex Market",
    "market": "Yandex Market",
    "yandex_market": "Yandex Market",
}


def sort_output_by_store(output: Path, stores: str):
    """Паук пишет товары по мере прихода, поэтому магазины перемешиваются.
    После завершения группируем строки по магазинам в порядке из --stores (внутри магазина порядок сохраняется)."""
    if not output.exists():
        return
    order = {}
    for name in (x.strip().lower() for x in stores.split(",") if x.strip()):
        order.setdefault(SOURCE_NAMES.get(name, name), len(order))

    def key(row):
        return order.get(row.get("source"), len(order))

    if output.suffix.lower() == ".json":
        rows = json.loads(output.read_text(encoding="utf-8"))
        rows.sort(key=key)
        output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    with output.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        fields = reader.fieldnames
        rows = list(reader)
    rows.sort(key=key)
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Run Lab 4 Scrapy spiders")
    parser.add_argument("--spider", required=True, choices=["news", "ecommerce"])
    parser.add_argument("--keywords")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--category", default="ноутбук")
    parser.add_argument("--stores", default="wildberries,ozon,yandexmarket")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-items", type=int, default=110)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    settings = get_project_settings()
    is_json = output.suffix.lower() == ".json"
    settings.set("FEEDS", {str(output): {
        "format": "json" if is_json else "csv",
        "encoding": "utf8" if is_json else "utf-8-sig",  # BOM, чтобы Excel открывал кириллицу и ₽
        "indent": 2 if is_json else None,
        "overwrite": True,
        "item_export_kwargs": {"delimiter": ";"} if not is_json else {},
    }})

    process = CrawlerProcess(settings)
    if args.spider == "news":
        from lab4_scraper.spiders.news import NewsSpider
        process.crawl(NewsSpider, keywords=args.keywords, days=args.days, max_items=args.max_items)
    else:
        from lab4_scraper.spiders.ecommerce import EcommerceSpider
        process.crawl(EcommerceSpider, category=args.category, stores=args.stores, max_items=args.max_items)
    process.start()

    if args.spider == "ecommerce":
        sort_output_by_store(output, args.stores)


if __name__ == "__main__":
    main()