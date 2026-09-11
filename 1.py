import argparse
import io
from urllib.parse import urlparse
from collections import Counter

import cdx_toolkit
import requests
from bs4 import BeautifulSoup
from tabulate import tabulate
from warcio.archiveiterator import ArchiveIterator


# заголовок
HEADERS = {
    "User-Agent": "CommonCrawlHomework/1.0 (educational project)"
}


# аргументы
def parse_args():
    parser = argparse.ArgumentParser(
        description="Поиск страниц в архиве Common Crawl"
    )
    parser.add_argument("keywords", nargs="+")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--show-text", action="store_true")

    return parser.parse_args()


# поиск в CDX
def search_cdx(domain, limit):
    print("Запрос CDX-индекса...")

    pattern = f"{domain}/*"

    try:
        cdx = cdx_toolkit.CDXFetcher(source="cc")

        # минимум 50 записей, потому что часть записей
        # может не подойти после фильтрации
        records = cdx.iter(
            pattern,
            limit=max(limit * 10, 50),
            filter=["=status:200"]
        )

        return [dict(record) for record in records]

    except Exception as e:
        print(f"Ошибка CDX: {e}")
        return []


# загрузка нужного фрагмента WARC
def load_warc(record):
    start = int(record["offset"])
    end = start + int(record["length"]) - 1

    url = "https://data.commoncrawl.org/" + record["filename"]

    try:
        # только нужные байты
        response = requests.get(
            url,
            headers={
                **HEADERS,
                "Range": f"bytes={start}-{end}"
            },
            timeout=60
        )

        response.raise_for_status()

        # Читаем WARC-запись
        for warc in ArchiveIterator(
            io.BytesIO(response.content)
        ):
            if warc.rec_type == "response":
                return warc

    except Exception as e:
        print(f"Ошибка WARC: {e}")

    return None


# получение заголовка и текста
def extract_page(warc):
    html = warc.content_stream().read()
    soup = BeautifulSoup(html, "html.parser")

    title = (
        soup.title.get_text(" ", strip=True)
        if soup.title
        else "-"
    )

    # лишние элементы
    for tag in soup([
        "script",
        "style",
        "noscript",
        "header",
        "footer",
        "nav"
    ]):
        tag.decompose()

    text = " ".join(
        soup.get_text(" ", strip=True).split()
    )

    return title, text


# проверка ключевых слов
def contains_keywords(text, keywords):
    text = text.lower()

    for keyword in keywords:
        if keyword.lower() not in text:
            return False

    return True


# считаем количество упоминаний
def count_mentions(text, keywords):
    text = text.lower()
    counts = {}

    for keyword in keywords:
        counts[keyword] = text.count(
            keyword.lower()
        )

    return counts


# небольшой фрагмент 300 символов
def make_fragment(text, keywords, size=300):
    lower = text.lower()

    for keyword in keywords:
        pos = lower.find(keyword.lower())

        if pos != -1:
            start = max(0, pos - 100)
            return text[start:start + size] + "..."

    return text[:size] + "..."


# перевод даты
def format_date(timestamp):
    return (
        f"{timestamp[6:8]}.{timestamp[4:6]}."
        f"{timestamp[:4]} {timestamp[8:10]}:"
        f"{timestamp[10:12]}:{timestamp[12:14]}"
    )


# получаем только дату
def get_short_date(timestamp):
    return (
        f"{timestamp[:4]}-"
        f"{timestamp[4:6]}-"
        f"{timestamp[6:8]}"
    )


# получаем домен из URL
def get_domain(url):
    return urlparse(url).netloc


# вывод статистики
def show_statistics(results, keywords):
    print()
    print("Статистика:")
    print(f"Найдено страниц: {len(results)}")

    # количество упоминаний каждого слова
    total_mentions = Counter()

    for result in results:
        for keyword, count in result["mentions"].items():
            total_mentions[keyword] += count

    print()
    print("Количество упоминаний:")

    for keyword in keywords:
        print(
            f"{keyword}: "
            f"{total_mentions[keyword]}"
        )

    # распределение по датам
    dates = Counter(
        result["short_date"]
        for result in results
    )

    print()
    print("По датам:")

    for date, count in sorted(dates.items()):
        print(f"{date} — {count}")

    # распределение по доменам
    domains = Counter(
        result["domain"]
        for result in results
    )

    print()
    print("По доменам:")

    for domain, count in domains.most_common():
        print(f"{domain} — {count}")


def main():
    args = parse_args()

    if args.limit <= 0:
        print("--limit должен быть больше нуля")
        return

    # получение записей CDX
    records = search_cdx(
        args.domain,
        args.limit
    )

    print(
        f"Получено записей CDX: "
        f"{len(records)}"
    )

    results = []

    for i, record in enumerate(records, 1):
        # проверка, что это HTML
        mime = (
            record.get("mime-detected")
            or record.get("mime")
            or ""
        )

        if "html" not in mime.lower():
            continue

        url = record["url"]
        date = format_date(record["timestamp"])

        if args.show_text:
            print(
                f"[{i}/{len(records)}] "
                f"Загрузка WARC: {url}"
            )

            # загрузка страницы из WARC
            warc = load_warc(record)

            if not warc:
                continue

            title, text = extract_page(warc)

            # проверяем ключевые слова
            if not contains_keywords(
                text,
                args.keywords
            ):
                continue

            # считаем упоминания
            mentions = count_mentions(
                text,
                args.keywords
            )

            # фрагмент текста
            fragment = make_fragment(
                text,
                args.keywords
            )

            results.append({
                "url": url,
                "date": date,
                "short_date": get_short_date(
                    record["timestamp"]
                ),
                "domain": get_domain(url),
                "title": title,
                "fragment": fragment,
                "mentions": mentions
            })

        else:
            results.append({
                "url": url,
                "date": date,
                "short_date": get_short_date(
                    record["timestamp"]
                ),
                "domain": get_domain(url),
                "title": "-",
                "fragment": "",
                "mentions": {}
            })

        if len(results) >= args.limit:
            break

    if not results:
        print("Страницы не найдены.")
        return

    # столбцы таблицы
    if args.show_text:
        table = [
            [
                result["url"],
                result["date"],
                result["title"],
                result["fragment"]
            ]
            for result in results
        ]

        headers = [
            "URL",
            "Дата архивации",
            "Заголовок",
            "Фрагмент текста"
        ]

        widths = [55, 20, 40, 60]

    else:
        table = [
            [
                result["url"],
                result["date"],
                result["title"]
            ]
            for result in results
        ]

        headers = [
            "URL",
            "Дата архивации",
            "Заголовок"
        ]

        widths = [60, 20, 40]

    # результаты
    print(
        tabulate(
            table,
            headers=headers,
            tablefmt="grid",
            maxcolwidths=widths
        )
    )

    # статистика
    if args.show_text:
        show_statistics(
            results,
            args.keywords
        )


if __name__ == "__main__":
    main()