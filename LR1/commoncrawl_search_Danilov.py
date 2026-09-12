import argparse
import io
import json
import math
import time
from collections import Counter
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from tabulate import tabulate
from warcio.archiveiterator import ArchiveIterator


HEADERS = {
    "User-Agent": "CommonCrawlHomework/1.0"
}

INDEX_SERVER = "https://index.commoncrawl.org"
COLLINFO_URL = f"{INDEX_SERVER}/collinfo.json"

REQUEST_TIMEOUT = 20

CDX_RETRIES = 3
CDX_RETRY_DELAY = 3

WARC_RETRIES = 3
WARC_RETRY_DELAY = 2

CRAWLS_TO_CHECK = 15

# сколько записей берём из одного краула
RECORDS_PER_CRAWL = 50


def parse_args():
    parser = argparse.ArgumentParser(
        description="Поиск страниц в архиве Common Crawl"
    )

    parser.add_argument(
        "keywords",
        nargs="+",
        help="ключевые слова для поиска"
    )

    parser.add_argument(
        "--domain",
        nargs="+",
        default=["pstu.ru"],
        help="один или несколько доменов"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="количество результатов на каждый домен"
    )

    parser.add_argument(
        "--show-text",
        action="store_true",
        help="показать фрагмент текста"
    )

    return parser.parse_args()


def normalize_domain(domain):
    domain = domain.strip()

    if domain.startswith("http://"):
        domain = domain[7:]

    if domain.startswith("https://"):
        domain = domain[8:]

    return domain.rstrip("/")


def get_crawls():
    try:
        response = requests.get(
            COLLINFO_URL,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        data = response.json()

        crawls = []

        for item in data:
            crawl_id = item.get("id")

            if crawl_id:
                crawls.append(crawl_id)

        return crawls[:CRAWLS_TO_CHECK]

    except requests.RequestException as error:
        print(f"Ошибка получения списка краулов: {error}")
        return []

    except (ValueError, json.JSONDecodeError) as error:
        print(f"Ошибка обработки списка краулов: {error}")
        return []


def search_crawl(crawl, domain, limit):
    url = f"{INDEX_SERVER}/{crawl}-index"

    params = {
        "url": f"{domain}/*",
        "output": "json",
        "filter": [
            "status:200",
            "mime:text/html"
        ],
        "limit": limit
    }

    last_error = None

    for attempt in range(1, CDX_RETRIES + 1):
        try:
            response = requests.get(
                url,
                params=params,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT
            )

            if response.status_code == 404:
                return []

            if response.status_code in (429, 500, 502, 503, 504):
                last_error = f"HTTP {response.status_code}"

                if attempt < CDX_RETRIES:
                    time.sleep(CDX_RETRY_DELAY * attempt)

                continue

            response.raise_for_status()

            records = []

            for line in response.text.splitlines():
                line = line.strip()

                if not line:
                    continue

                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

            return records

        except requests.Timeout as error:
            last_error = f"таймаут: {error}"

        except requests.ConnectionError as error:
            last_error = f"ошибка соединения: {error}"

        except requests.RequestException as error:
            last_error = str(error)

        if attempt < CDX_RETRIES:
            time.sleep(CDX_RETRY_DELAY * attempt)

    print(
        f"    не удалось получить данные после "
        f"{CDX_RETRIES} попыток: {last_error}"
    )

    return []


def search_cdx(domains, limit):
    print()
    print("Получение списка краулов...")

    crawls = get_crawls()

    if not crawls:
        return []

    print(f"Найдено краулов для проверки: {len(crawls)}")

    for crawl in crawls:
        print(f"  {crawl}")

    print()

    # берём запас записей, потому что часть страниц не подойдёт
    records_per_crawl = max(
        RECORDS_PER_CRAWL,
        math.ceil(limit * 3 / len(domains))
    )

    print(
        f"Записей на домен в одном крауле: "
        f"{records_per_crawl}"
    )

    all_records = []

    for domain_index, domain in enumerate(domains, 1):
        domain = normalize_domain(domain)

        print()
        print(
            f"[ДОМЕН {domain_index}/{len(domains)}] "
            f"{domain}"
        )

        for crawl_index, crawl in enumerate(crawls, 1):
            print(
                f"  [{crawl_index}/{len(crawls)}] "
                f"{crawl}...",
                end=" ",
                flush=True
            )

            records = search_crawl(
                crawl,
                domain,
                records_per_crawl
            )

            print(f"получено: {len(records)}")

            # запоминаем, откуда пришла запись
            for record in records:
                record["_search_domain"] = domain

            all_records.extend(records)

            # не спамим Common Crawl запросами
            time.sleep(0.5)

    print()

    return all_records


def remove_duplicates(records):
    unique = {}

    for record in records:
        key = (
            record.get("url"),
            record.get("timestamp"),
            record.get("digest")
        )

        unique[key] = record

    return list(unique.values())


def load_warc(record):
    start = int(record["offset"])
    end = start + int(record["length"]) - 1

    url = (
        "https://data.commoncrawl.org/"
        + record["filename"]
    )

    headers = {
        **HEADERS,
        "Range": f"bytes={start}-{end}"
    }

    last_error = None

    for attempt in range(1, WARC_RETRIES + 1):
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=REQUEST_TIMEOUT
            )

            response.raise_for_status()

            for warc in ArchiveIterator(
                io.BytesIO(response.content)
            ):
                if warc.rec_type == "response":
                    return warc

            return None

        except requests.Timeout as error:
            last_error = f"таймаут: {error}"

        except requests.ConnectionError as error:
            last_error = f"ошибка соединения: {error}"

        except requests.RequestException as error:
            last_error = str(error)

        except Exception as error:
            last_error = str(error)
            break

        if attempt < WARC_RETRIES:
            time.sleep(WARC_RETRY_DELAY * attempt)

    print(
        f"    ошибка WARC после "
        f"{WARC_RETRIES} попыток: "
        f"{last_error}"
    )

    return None


def extract_page(warc):
    try:
        html = warc.content_stream().read()

        soup = BeautifulSoup(
            html,
            "html.parser"
        )

        title = (
            soup.title.get_text(
                " ",
                strip=True
            )
            if soup.title
            else "-"
        )

        # убираем то, что не относится к тексту страницы
        for tag in soup(
            [
                "script",
                "style",
                "noscript",
                "header",
                "footer",
                "nav"
            ]
        ):
            tag.decompose()

        text = " ".join(
            soup.get_text(
                " ",
                strip=True
            ).split()
        )

        return title, text

    except Exception as error:
        print(
            f"    ошибка обработки страницы: "
            f"{error}"
        )

        return "-", ""


def contains_keywords(text, keywords):
    text = text.lower()

    # страница подходит только при наличии всех слов
    return all(
        keyword.lower() in text
        for keyword in keywords
    )


def count_mentions(text, keywords):
    text = text.lower()

    return {
        keyword: text.count(
            keyword.lower()
        )
        for keyword in keywords
    }


def make_fragment(text, keywords, size=300):
    text_lower = text.lower()

    for keyword in keywords:
        position = text_lower.find(
            keyword.lower()
        )

        if position == -1:
            continue

        start = max(
            0,
            position - 100
        )

        fragment = text[
            start:start + size
        ]

        if start > 0:
            fragment = "..." + fragment

        if start + size < len(text):
            fragment += "..."

        return fragment

    return text[:size] + "..."


def format_date(timestamp):
    return (
        f"{timestamp[6:8]}."
        f"{timestamp[4:6]}."
        f"{timestamp[:4]} "
        f"{timestamp[8:10]}:"
        f"{timestamp[10:12]}:"
        f"{timestamp[12:14]}"
    )


def get_short_date(timestamp):
    return (
        f"{timestamp[:4]}-"
        f"{timestamp[4:6]}-"
        f"{timestamp[6:8]}"
    )


def get_domain(url):
    return urlparse(url).netloc


def print_results(results, show_text):
    if not results:
        print()
        print("Подходящих страниц не найдено.")
        return

    table = []

    for result in results:
        row = [
            result["url"],
            result["date"],
            result["title"]
        ]

        if show_text:
            row.append(result["fragment"])

        table.append(row)

    headers = [
        "URL",
        "Дата архива",
        "Заголовок"
    ]

    if show_text:
        headers.append("Фрагмент текста")

    widths = (
        [60, 20, 50, 80]
        if show_text
        else [60, 20, 50]
    )

    print()
    print("РЕЗУЛЬТАТЫ ПОИСКА")

    print(
        tabulate(
            table,
            headers=headers,
            tablefmt="grid",
            maxcolwidths=widths
        )
    )


def show_statistics(results, keywords, domains):
    print()
    print("СТАТИСТИКА")

    print(
        f"Найдено подходящих страниц: "
        f"{len(results)}"
    )

    mentions = Counter()

    for result in results:
        mentions.update(
            result["mentions"]
        )

    print()
    print("Количество упоминаний:")

    for keyword in keywords:
        print(
            f"  {keyword}: "
            f"{mentions[keyword]}"
        )

    dates = Counter(
        result["short_date"]
        for result in results
    )

    print()
    print("По датам:")

    for date, count in sorted(
        dates.items()
    ):
        print(
            f"  {date} — {count}"
        )

    domain_counts = Counter(
        result["domain"]
        for result in results
    )

    print()
    print("По доменам:")

    # показываем именно те домены, которые указал пользователь
    for domain in domains:
        domain = normalize_domain(domain)

        print(
            f"  {domain} — "
            f"{domain_counts.get(domain, 0)}"
        )


def main():
    args = parse_args()

    if args.limit <= 0:
        print(
            "Ошибка: --limit должен быть "
            "больше нуля."
        )
        return

    domains = [
        normalize_domain(domain)
        for domain in args.domain
    ]

    print("COMMON CRAWL SEARCH")

    print(
        "Ключевые слова: "
        + ", ".join(args.keywords)
    )

    print(
        "Домены: "
        + ", ".join(domains)
    )

    print(
        f"Лимит результатов на домен: "
        f"{args.limit}"
    )

    print(
        "Показ текста: "
        + (
            "да"
            if args.show_text
            else "нет"
        )
    )

    records = search_cdx(
        domains,
        args.limit
    )

    if not records:
        print()
        print("Записи Common Crawl не найдены.")
        return

    records = remove_duplicates(
        records
    )

    # сначала проверяем свежие записи
    records.sort(
        key=lambda record: record.get(
            "timestamp",
            ""
        ),
        reverse=True
    )

    print(
        f"Всего уникальных записей CDX: "
        f"{len(records)}"
    )

    print()
    print("ПРОВЕРКА СОДЕРЖИМОГО")

    results = []

    # считаем результаты отдельно для каждого домена
    domain_results = {
        domain: []
        for domain in domains
    }

    for index, record in enumerate(
        records,
        1
    ):
        url = record.get("url")
        timestamp = record.get("timestamp")

        if not url or not timestamp:
            continue

        domain = get_domain(url)

        if domain not in domain_results:
            continue

        # не ищем больше нужного количества страниц
        if len(domain_results[domain]) >= args.limit:
            continue

        print(
            f"[{index}/{len(records)}] "
            f"{url}"
        )

        warc = load_warc(record)

        if not warc:
            continue

        title, text = extract_page(
            warc
        )

        if not contains_keywords(
            text,
            args.keywords
        ):
            continue

        mentions = count_mentions(
            text,
            args.keywords
        )

        fragment = ""

        if args.show_text:
            fragment = make_fragment(
                text,
                args.keywords
            )

        result = {
            "url": url,
            "date": format_date(timestamp),
            "short_date": get_short_date(timestamp),
            "domain": domain,
            "title": title,
            "fragment": fragment,
            "mentions": mentions
        }

        domain_results[domain].append(
            result
        )

        results.append(result)

        print(
            f"    найдено совпадений: "
            f"{len(domain_results[domain])} "
            f"для {domain}"
        )

        # заканчиваем поиск, когда лимит набран для всех доменов
        if all(
            len(domain_results[domain]) >= args.limit
            for domain in domains
        ):
            break

    # собираем результаты в порядке доменов
    displayed_results = []

    for domain in domains:
        displayed_results.extend(
            domain_results[domain]
        )

    print_results(
        displayed_results,
        args.show_text
    )

    show_statistics(
        displayed_results,
        args.keywords,
        domains
    )


if __name__ == "__main__":
    main()