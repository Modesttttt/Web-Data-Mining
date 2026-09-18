from __future__ import annotations

import argparse
import csv
import inspect
import json
import logging
import re
import time
from collections import Counter, namedtuple
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from graphviz import Digraph
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

if not hasattr(inspect, "getargspec"):
    ArgSpec = namedtuple("ArgSpec", ["args", "varargs", "keywords", "defaults"])

    def getargspec(func):
        spec = inspect.getfullargspec(func)
        return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = getargspec

from rutermextract import TermExtractor

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "resources.csv"
RESULTS = ROOT / "results"

DEFAULT_THRESHOLD = 3
DEFAULT_MAX_TERMS = 60
DEFAULT_DELAY = 0.35
TIMEOUT = 20


# типичный веб-шум, который не характеризует металлургию.
BOILERPLATE_TERMS = {
    "компания", "контакты", "контакт", "информация", "документы", "новости",
    "развитие", "деятельность", "работа", "участие", "поддержка", "управление",
    "сотрудники", "карьера", "поставщики", "инвесторы", "клиенты", "партнёры",
    "партнеры", "проект", "проекты", "услуги", "год", "город", "мир", "область",
    "страна", "россия", "казахстан", "узбекистан", "группа", "система", "раздел",
    "страница", "сайт", "главная", "пресс-центр", "пресс-релизы", "политика",
    "персональные данные", "файлы cookie", "cookie", "конфиденциальность",
    "обратная связь", "реквизиты", "вакансии", "вакансия", "руководство",
}
BOILERPLATE_PATTERNS = (
    "персональн", "cookie", "конфиденциаль", "google analytics", "яндекс метрик",
    "политика конфиденциальности", "обработка данных",
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("metallurgy-graph")


@dataclass
class ResourceResult:
    id: int
    name: str
    country: str
    url: str
    status_code: int | None
    title: str
    text_length: int
    terms: list[str]
    error: str | None = None


def load_resources() -> list[dict]:
    with DATA_FILE.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({
        "User-Agent": "Web-Data-Mining-LR1/1.0 (educational project)",
        "Accept-Language": "ru,en;q=0.8",
    })
    return session


def extract_main_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg", "template"]):
        tag.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    nodes = soup.select("p, h1, h2, h3, h4, h5, h6, li")
    parts = []
    for node in nodes:
        text = " ".join(node.stripped_strings)
        if text:
            parts.append(text)

    text = "\n".join(parts)
    text = re.sub(r"\s+", " ", text).strip()
    return title, text


def extract_terms(text: str, extractor: TermExtractor, max_terms: int) -> list[str]:
    if not text:
        return []

    text = text[:250_000]
    terms = extractor(text)
    normalized = []
    seen = set()

    for term in terms:
        value = getattr(term, "normalized", None) or getattr(term, "term", None)
        if not value:
            continue
        value = re.sub(r"\s+", " ", value.strip().lower())
        if len(value) < 3 or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
        if len(normalized) >= max_terms:
            break

    return normalized


def is_boilerplate(term: str) -> bool:
    if term in BOILERPLATE_TERMS:
        return True
    return any(pattern in term for pattern in BOILERPLATE_PATTERNS)

# удаление веб-шаблонов и слишком частых терминов, с сохранением тематических.
def clean_terms(results: list[ResourceResult], max_terms: int) -> None:
    frequency = Counter(
        term
        for result in results
        for term in set(result.terms)
        if not is_boilerplate(term)
    )

    resource_count = sum(1 for result in results if result.error is None)
    # термин, встречающийся почти на всех страницах, обычно является шаблоном сайта.
    frequent_limit = max(3, int(resource_count * 0.65))

    for result in results:
        if result.error is not None:
            continue

        filtered = [
            term for term in result.terms
            if not is_boilerplate(term) and frequency[term] <= frequent_limit
        ]
        result.terms = filtered[:max_terms]


def collect(resources: list[dict], delay: float, max_terms: int) -> list[ResourceResult]:
    session = build_session()
    extractor = TermExtractor()
    results: list[ResourceResult] = []

    for index, resource in enumerate(resources, start=1):
        log.info("[%02d/%02d] %s", index, len(resources), resource["name"])
        status = None
        title = ""
        text = ""
        error = None
        try:
            response = session.get(resource["url"], timeout=TIMEOUT, allow_redirects=True)
            status = response.status_code
            response.raise_for_status()
            response.encoding = response.apparent_encoding or response.encoding
            title, text = extract_main_text(response.text)
            terms = extract_terms(text, extractor, max_terms)
            log.info("  status=%s text=%d chars terms=%d", status, len(text), len(terms))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            terms = []
            log.warning("  skipped: %s", error)

        results.append(ResourceResult(
            id=int(resource["id"]),
            name=resource["name"],
            country=resource["country"],
            url=resource["url"],
            status_code=status,
            title=title,
            text_length=len(text),
            terms=terms,
            error=error,
        ))
        time.sleep(delay)

    clean_terms(results, max_terms)
    return results


def save_json(results: list[ResourceResult]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = [asdict(item) for item in results]
    (RESULTS / "resources.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_edges(results: list[ResourceResult], threshold: int):
    edges = []
    for left, right in combinations(results, 2):
        if left.error is not None or right.error is not None:
            continue
        common = sorted(set(left.terms) & set(right.terms))
        if len(common) >= threshold:
            edges.append({
                "source": left.id,
                "target": right.id,
                "common_terms": common,
                "weight": len(common),
            })
    edges.sort(key=lambda x: (-x["weight"], x["source"], x["target"]))
    return edges


def render_graph(results: list[ResourceResult], edges: list[dict]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    dot = Digraph("metallurgy_resources", format="svg")
    dot.attr(rankdir="LR", splines="true", overlap="false")
    dot.attr("node", shape="box", style="rounded", fontname="Arial", fontsize="10")
    dot.attr("edge", fontname="Arial", fontsize="8")

    for item in results:
        label = f"{item.id}. {item.name}"
        dot.node(str(item.id), label)

    for edge in edges:
        dot.edge(
            str(edge["source"]),
            str(edge["target"]),
            label=str(edge["weight"]),
            tooltip=", ".join(edge["common_terms"][:12]),
        )

    dot.save(str(RESULTS / "graph.dot"))
    dot.render(filename=str(RESULTS / "metallurgy_graph"), cleanup=True)

    (RESULTS / "edges.json").write_text(
        json.dumps(edges, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def print_summary(results: list[ResourceResult], edges: list[dict], threshold: int) -> None:
    ok = [r for r in results if r.error is None]
    with_terms = [r for r in ok if r.terms]
    print("\nРЕЗУЛЬТАТ:")
    print(f"Ресурсов в списке: {len(results)}")
    print(f"Успешно обработано: {len(ok)}")
    print(f"Ресурсов с ошибками: {len(results) - len(ok)}")
    print(f"Ресурсов с терминами: {len(with_terms)}")
    print(f"Порог общего количества терминов: {threshold}")
    print(f"Рёбер графа: {len(edges)}")
    print(f"DOT: {RESULTS / 'graph.dot'}")
    print(f"SVG: {RESULTS / 'metallurgy_graph.svg'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ЛР1: граф металлургических информационных ресурсов"
    )
    parser.add_argument(
        "--threshold", type=int, default=DEFAULT_THRESHOLD,
        help="минимальное число общих терминов для ребра"
    )
    parser.add_argument(
        "--delay", type=float, default=DEFAULT_DELAY,
        help="пауза между запросами, секунд"
    )
    parser.add_argument(
        "--max-terms", type=int, default=DEFAULT_MAX_TERMS,
        help="максимум терминов на ресурс"
    )
    args = parser.parse_args()

    if args.threshold < 1:
        raise SystemExit("--threshold должен быть >= 1")

    resources = load_resources()
    if len(resources) < 25:
        raise SystemExit("В data/resources.csv должно быть минимум 25 ресурсов")

    results = collect(resources, args.delay, args.max_terms)
    save_json(results)
    edges = build_edges(results, args.threshold)
    render_graph(results, edges)
    print_summary(results, edges, args.threshold)


if __name__ == "__main__":
    main()
