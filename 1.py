import argparse
import io

import cdx_toolkit
import requests
from bs4 import BeautifulSoup
from tabulate import tabulate
from warcio.archiveiterator import ArchiveIterator


# заголовок
HEADERS = {
    "User-Agent": "CommonCrawlModesttttt"
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

        # чтение WARC
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

    # лишние элементы отбрасываем
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


# небольшой фрагмент текста (300 символов)
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

    print(f"Получено записей CDX: {len(records)}")

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

            # фрагмент текста
            fragment = make_fragment(
                text,
                args.keywords
            )

            results.append([
                url,
                date,
                title,
                fragment
            ])

        else:
            results.append([
                url,
                date,
                "-"
            ])

        if len(results) >= args.limit:
            break

    if not results:
        print("Страницы не найдены.")
        return

    # столбцы таблицы
    if args.show_text:
        headers = [
            "URL",
            "Дата архивации",
            "Заголовок",
            "Фрагмент текста"
        ]
        widths = [55, 20, 40, 60]
    else:
        headers = [
            "URL",
            "Дата архивации",
            "Заголовок"
        ]
        widths = [60, 20, 40]

    print(
        tabulate(
            results,
            headers=headers,
            tablefmt="grid",
            maxcolwidths=widths
        )
    )


if __name__ == "__main__":
    main()