import json
import re
import time
from datetime import date
from pathlib import Path
from urllib.parse import quote

import matplotlib.pyplot as plt
import requests

QUERY = "искусственный интеллект"
N_ARTICLES = 30
LINK_LIMIT = 100
BATCH = 5
OUT = Path("wikimedia_lab3_output")
OUT.mkdir(exist_ok=True)

WIKI = "https://ru.wikipedia.org/w/api.php"
WD = "https://www.wikidata.org/w/api.php"
COMMONS = "https://commons.wikimedia.org/w/api.php"
WIKTIONARY = "https://ru.wiktionary.org/w/api.php"

S = requests.Session()
S.headers["User-Agent"] = "University-Lab3-WikimediaAPI/3.0 (educational project)"


def api(url, params):
    for attempt in range(5):
        try:
            r = S.get(url, params={**params, "format": "json", "formatversion": 2}, timeout=25)
            if r.status_code == 429:
                value = r.headers.get("Retry-After", "10")
                try:
                    wait = min(float(value), 60)
                except ValueError:
                    wait = min(10 * (attempt + 1), 60)
                print(f"    429: повтор через {wait:.0f} с...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            time.sleep(0.45)
            return r.json()
        except (requests.RequestException, ValueError) as e:
            if attempt == 4:
                raise
            wait = min(3 * (attempt + 1), 15)
            print(f"    Ошибка API: {e}; повтор через {wait} с...")
            time.sleep(wait)
    raise RuntimeError("API недоступен")


def clean(value):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", str(value or ""))).strip()


def wiki_url(title):
    return "https://ru.wikipedia.org/wiki/" + quote(title.replace(" ", "_"))


def search_articles():
    """
    Вместо того чтобы просто брать первые N результатов одного
    поиска, собираем кандидатов из нескольких связанных запросов,
    рассчитываем тематический балл по заголовку и краткому описанию
    и исключаем явно посторонние страницы.
    """

    search_queries = [
        "искусственный интеллект",
        "машинное обучение",
        "нейронная сеть",
        "генеративный искусственный интеллект",
        "большая языковая модель",
    ]

    # Термины, по которым статья считается тематически связанной
    # с выбранной предметной областью.
    strong_terms = (
        "искусственный интеллект",
        "искусственный разум",
        "машинное обучение",
        "нейронн",
        "генеративн",
        "языковая модель",
        "большая языковая модель",
        "глубокое обучение",
        "компьютерное зрение",
        "обработка естественного языка",
        "чат-бот",
        "робот",
        "трансформер",
        "нейросет",
        "ai",
        "ии",
        "gpt",
        "llm",
        "openai",
        "anthropic",
        "deepseek",
        "gemini",
        "elevenlabs",
        "cursor",
    )

    # Тематические слова, которые особенно хорошо характеризуют
    # статью как материал непосредственно про ИИ.
    title_terms = (
        "искусственный интеллект",
        "машинное обучение",
        "нейронная сеть",
        "нейросеть",
        "генеративный искусственный интеллект",
        "языковая модель",
        "большая языковая модель",
        "глубокое обучение",
        "чат-бот",
        "искусственный разум",
        "робот",
        "трансформер",
        "компьютерное зрение",
        "обработка естественного языка",
        "gpt",
        "llm",
        "openai",
        "anthropic",
        "deepseek",
        "gemini",
        "elevenlabs",
    )

    # Явное исключение страниц, которые могут попасть в поиск
    # из-за косвенного совпадения, но не являются хорошими
    # представителями предметной области.
    bad_terms = (
        "значения",
        "список",
        "фильм",
        "телесериал",
        "мультфильм",
        "песня",
        "альбом",
        "роман",
        "рассказ",
        "игра (",
        "bigo live",
        "социальная сеть",
        "телеканал",
        "газета",
        "журнал",
        "футбольный клуб",
        "хоккейный клуб",
        "автомобиль",
        "самолёт",
        "корабль",
        "станция метро",
        "населённый пункт",
    )

    candidates = {}

    for query in search_queries:
        data = api(
            WIKI,
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srnamespace": 0,
                "srlimit": 100,
            }
        )

        for item in data.get(
            "query", {}
        ).get("search", []):

            title = item.get("title", "").strip()

            if not title or ":" in title:
                continue

            low_title = title.casefold()
            snippet = clean(
                item.get("snippet", "")
            )
            low_text = (
                f"{low_title} {snippet.casefold()}"
            )

            if any(
                bad in low_title
                for bad in bad_terms
            ):
                continue

            # Базовый балл — наличие тематических терминов
            # в заголовке и кратком описании.
            title_score = sum(
                12
                for term in title_terms
                if term in low_title
            )

            text_score = sum(
                3
                for term in strong_terms
                if term in low_text
            )

            # Для названий с прямым упоминанием ИИ добавляем
            # дополнительный вес.
            if "искусственный интеллект" in low_title:
                title_score += 30

            if "машинное обучение" in low_title:
                title_score += 28

            if "нейрон" in low_title:
                title_score += 25

            if (
                "языковая модель" in low_title
                or "большая языковая модель" in low_title
            ):
                title_score += 25

            if "генератив" in low_title:
                title_score += 22

            if (
                "chatgpt" in low_title
                or "gpt-" in low_title
                or low_title == "openai"
                or "anthropic" in low_title
                or "deepseek" in low_title
                or "gemini" in low_title
            ):
                title_score += 20

            # Чем выше позиция в выдаче Wikimedia, тем небольшой
            # дополнительный бонус получает кандидат.
            rank_bonus = max(
                0,
                15 - item.get("index", 15)
            )

            score = title_score + text_score + rank_bonus

            # Кандидат должен иметь хотя бы один явный тематический
            # признак.
            if not any(
                term in low_text
                for term in strong_terms
            ):
                continue

            key = low_title

            candidate = {
                "score": score,
                "title": title,
                "snippet": snippet,
            }

            # Если статья встретилась в нескольких поисковых запросах,
            # сохраняем максимальный набранный балл.
            if (
                key not in candidates
                or score > candidates[key]["score"]
            ):
                candidates[key] = candidate

    ranked = sorted(
        candidates.values(),
        key=lambda x: (
            -x["score"],
            x["title"].casefold()
        )
    )

    # Берём только действительно тематические страницы.
    selected = ranked[:N_ARTICLES]

    if len(selected) < N_ARTICLES:
        raise RuntimeError(
            "После тематической фильтрации найдено только "
            f"{len(selected)} подходящих статей из "
            f"требуемых {N_ARTICLES}"
        )

    print(
        f"  Кандидатов после фильтрации: "
        f"{len(ranked)}"
    )

    print("  Выбранные статьи:")

    for i, item in enumerate(selected, 1):
        print(
            f"    {i:02d}. {item['title']} "
            f"(балл: {item['score']})"
        )

    return [
        item["title"]
        for item in selected
    ]


def get_pages(titles):
    pages = {}
    for start in range(0, len(titles), BATCH):
        batch = titles[start:start + BATCH]
        data = api(WIKI, {
            "action": "query",
            "prop": "extracts|info|pageprops",
            "titles": "|".join(batch),
            "exintro": 1, "explaintext": 1, "inprop": "url"
        })
        for p in data.get("query", {}).get("pages", []):
            if not p.get("pageid"):
                continue
            title = p.get("title", "")
            pages[title] = {
                "title": title,
                "pageid": p["pageid"],
                "url": p.get("fullurl") or wiki_url(title),
                "extract": clean(p.get("extract")),
                "links": [],
                "images": [],
                "wikidata_id": p.get("pageprops", {}).get("wikibase_item")
            }
    return [pages[t] for t in titles if t in pages]


def get_links(articles):
    # pllimit задаёт общий лимит для одного API-запроса,
    # а не отдельный лимит для каждой страницы. Поэтому запрос
    # ссылки для каждой статьи отдельно, чтобы данные не "съедались"
    # предыдущими страницами в batch-запросе.
    for i, article in enumerate(articles, 1):
        data = api(WIKI, {
            "action": "query",
            "prop": "links",
            "titles": article["title"],
            "plnamespace": 0,
            "pllimit": LINK_LIMIT
        })

        pages = data.get("query", {}).get("pages", [])
        links = []

        if pages:
            for x in pages[0].get("links", []):
                t = x.get("title", "")
                if t and ":" not in t and t != article["title"]:
                    links.append({
                        "title": t,
                        "url": wiki_url(t)
                    })

        article["links"] = links
        print(
            f"  {i:02d}/{len(articles)}  "
            f"{article['title']} | ссылок: {len(links)}"
        )

    return articles

def get_wikidata(articles):
    ids = [a["wikidata_id"] for a in articles if a.get("wikidata_id")]
    result = []
    for start in range(0, len(ids), 50):
        data = api(WD, {
            "action": "wbgetentities",
            "ids": "|".join(ids[start:start + 50]),
            "languages": "ru|en",
            "props": "labels|descriptions|sitelinks"
        })
        for qid, e in data.get("entities", {}).items():
            result.append({
                "id": qid,
                "label_ru": e.get("labels", {}).get("ru", {}).get("value"),
                "label_en": e.get("labels", {}).get("en", {}).get("value"),
                "description_ru": e.get("descriptions", {}).get("ru", {}).get("value"),
                "sitelinks_count": len(e.get("sitelinks", {}))
            })
    return result


def get_commons():
    data = api(COMMONS, {
        "action": "query", "generator": "search",
        "gsrsearch": QUERY, "gsrnamespace": 6, "gsrlimit": 5,
        "prop": "imageinfo", "iiprop": "url|extmetadata"
    })
    result = []
    for p in data.get("query", {}).get("pages", []):
        info = (p.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata", {})
        result.append({
            "title": p.get("title"),
            "url": info.get("url"),
            "description": clean(meta.get("ImageDescription", {}).get("value"))
        })
    return result


def get_wiktionary():
    term = "искусственный"
    data = api(WIKTIONARY, {
        "action": "query", "prop": "extracts|info", "titles": term,
        "exintro": 1, "explaintext": 1, "inprop": "url"
    })
    p = (data.get("query", {}).get("pages") or [{}])[0]
    return {
        "title": p.get("title", term),
        "pageid": p.get("pageid"),
        "url": p.get("fullurl"),
        "extract": clean(p.get("extract"))
    }


def get_potd():
    day = date.today().strftime("%Y-%m-%d")
    data = api(COMMONS, {
        "action": "parse", "page": f"Template:Potd/{day}",
        "prop": "wikitext|images"
    })
    parsed = data.get("parse", {})
    filename = next(
        (x for x in parsed.get("images", [])
         if x.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"))
         and not any(b in x.lower() for b in ("logo", "icon", "notification", "commons"))),
        None
    )

    if not filename:
        text = parsed.get("wikitext", "")
        if isinstance(text, dict):
            text = text.get("*", "")
        match = re.search(r"\[\[(?:File|Файл):([^|\]\n]+)", str(text), re.I)
        if match:
            filename = match.group(1).strip()

    if not filename:
        raise RuntimeError("Файл Picture of the Day не найден")

    filename = filename if filename.lower().startswith("file:") else "File:" + filename
    data = api(COMMONS, {
        "action": "query", "titles": filename, "prop": "imageinfo",
        "iiprop": "url|extmetadata"
    })
    p = (data.get("query", {}).get("pages") or [{}])[0]
    info = (p.get("imageinfo") or [{}])[0]
    meta = info.get("extmetadata", {})

    return {
        "date": day,
        "title": filename,
        "image_url": info.get("url"),
        "description": clean(
            meta.get("ImageDescription", {}).get("value")
            or meta.get("ObjectName", {}).get("value")
        ),
        "source_url": "https://commons.wikimedia.org/wiki/" +
                      quote(filename.replace(" ", "_"))
    }


def graph_edges(articles):
    titles = {a["title"] for a in articles}
    return sorted({
        (a["title"], link["title"])
        for a in articles
        for link in a["links"]
        if link["title"] in titles and link["title"] != a["title"]
    })


def save_dot(articles, edges):
    esc = lambda s: s.replace("\\", "\\\\").replace('"', '\\"')
    lines = [
        "digraph WikimediaKnowledgeGraph {", "    rankdir=LR;",
        '    graph [label="Граф связей Wikimedia", labelloc="t", fontsize=18];',
        '    node [shape=box, style="rounded"];'
    ]
    lines += [f'    "{esc(a["title"])}";' for a in articles]
    lines += [f'    "{esc(s)}" -> "{esc(t)}";' for s, t in edges]
    lines.append("}")
    path = OUT / "wikimedia_graph.dot"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def save_plot(articles, edges):
    degree = {a["title"]: 0 for a in articles}
    for s, t in edges:
        degree[s] += 1
        degree[t] += 1

    rows = sorted(degree.items(), key=lambda x: (-x[1], x[0].casefold()))
    names, values = zip(*rows)

    plt.figure(figsize=(13, max(8, len(names) * 0.32)))
    bars = plt.barh(names, values)
    plt.gca().invert_yaxis()
    plt.title("Связанность собранных статей Википедии")
    plt.xlabel("Количество связей с другими собранными статьями")
    plt.ylabel("Статьи Википедии")
    plt.grid(axis="x", alpha=0.25)

    for bar, value in zip(bars, values):
        if value:
            plt.text(value + 0.1, bar.get_y() + bar.get_height() / 2,
                     str(value), va="center")

    plt.tight_layout()
    path = OUT / "links_statistics.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def main():
    print("=" * 70)
    print("ЛАБОРАТОРНАЯ РАБОТА №3 — WIKIMEDIA API")
    print("=" * 70)
    print(f"Предметная область: {QUERY}")
    print(f"Целевое количество статей: {N_ARTICLES}")

    print("\n[1/6] Сбор статей Википедии")
    titles = search_articles()
    articles = get_pages(titles)
    if len(articles) < N_ARTICLES:
        raise RuntimeError(f"Получено только {len(articles)} страниц")

    articles = get_links(articles)
    for i, a in enumerate(articles, 1):
        print(f"  {i:02d}/{N_ARTICLES}  {a['title']} | ссылок: {len(a['links'])}")

    print("\n[2/6] Получение данных Wikidata")
    wikidata = get_wikidata(articles)
    print(f"  Найдено сущностей Wikidata: {len(wikidata)}")

    print("\n[3/6] Поиск изображений Wikimedia Commons")
    try:
        commons = get_commons()
        print(f"  Найдено файлов: {len(commons)}")
    except Exception as e:
        commons = []
        print(f"  Предупреждение: {e}")

    print("\n[4/6] Получение данных Викисловаря")
    try:
        wiktionary = get_wiktionary()
        print(f"  Термин: {wiktionary['title']}")
        print(f"  Страница: {wiktionary['url']}")
    except Exception as e:
        wiktionary = {"title": "искусственный", "error": str(e)}
        print(f"  Предупреждение: {e}")

    print("\n[5/6] Построение графа связей")
    edges = graph_edges(articles)
    print(f"  Узлов: {len(articles)}")
    print(f"  Рёбер: {len(edges)}")

    try:
        potd = get_potd()
        safe = re.sub(r'[\\/:*?"<>|]+', "_", potd["title"].replace("File:", ""))
        potd_path = OUT / f"{safe}.txt"
        potd_path.write_text(
            f"Название: {potd['title']}\nИсточник: {potd['source_url']}\n\n"
            f"Описание:\n{potd['description']}\n", encoding="utf-8"
        )
        print("\nPicture of the Day:")
        print(f"  Файл: {potd['title']}")
        print(f"  Описание сохранено: {potd_path}")
    except Exception as e:
        potd = None
        print(f"\n  Предупреждение POTD: {e}")

    result = {
        "metadata": {
            "laboratory": "Лабораторная работа №3",
            "subject_area": QUERY,
            "article_limit": N_ARTICLES,
            "articles_collected": len(articles)
        },
        "wikipedia": articles,
        "wikidata": wikidata,
        "wikimedia_commons": commons,
        "wiktionary": wiktionary,
        "graph": {
            "nodes": [a["title"] for a in articles],
            "edges": [{"source": s, "target": t} for s, t in edges]
        }
    }
    if potd:
        result["picture_of_the_day"] = potd

    json_path = OUT / "wikimedia_result.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    dot_path = save_dot(articles, edges)
    plot_path = save_plot(articles, edges)

    print("\n[6/6] Проверка требований")
    checks = [
        ("30 статей", len(articles) >= 30),
        ("JSON", json_path.exists()),
        ("Wikipedia", bool(articles)),
        ("Wikidata", bool(wikidata)),
        ("Commons", bool(commons)),
        ("Викисловарь", bool(wiktionary)),
        ("Picture of the Day", potd is not None),
        ("DOT", dot_path.exists()),
        ("30 узлов", len(articles) >= 30),
        ("Есть рёбра", len(edges) >= 10)
    ]

    ok = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok &= passed

    print("РЕЗУЛЬТАТ")
    print(" ")
    print(f"Статей:       {len(articles)}")
    print(f"Узлов графа:  {len(articles)}")
    print(f"Рёбер графа:  {len(edges)}")
    print(f"Wikidata:     {len(wikidata)}")
    print(f"Commons:      {len(commons)}")
    print(f"JSON:         {json_path}")
    print(f"DOT:          {dot_path}")
    print(f"График:       {plot_path}")
    print(f"Проверки:     {'ВСЕ ПРОЙДЕНЫ' if ok else 'ЕСТЬ ПРОБЛЕМЫ'}")



if __name__ == "__main__":
    main()
