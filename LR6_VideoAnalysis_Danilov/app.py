# веб-интерфейс на Gradio: ищу моменты в роликах и открываю их с нужной секунды

import html
import re
from collections import Counter

from common import (
    EVENT_TYPES_RU, OBJECT_NAMES_RU, RESULTS_DIR, SCENE_MIN_SCORE, YOLO_CONF, ensure_cwd, fmt_time,
    load_index, read_csv, youtube_link,
)
from entities import find_entities

SCENE_COLORS = {
    "терминал": "#2f3b45", "рабочий стол": "#3b6ea8", "установщик ОС": "#e08a2c",
    "настройки": "#8a6bbf", "браузер": "#3a9d8f", "человек в кадре": "#d1495b",
    "игра": "#5bb04a", "заставка": "#c9a227",
}
TRACK_COLORS = {"object": "#7a8b99", "scene_cut": "#222222", "loud": "#e4572e",
                "silence": "#9db4c0", "mention": "#6a4c93", "chapter": "#c9a227"}
SEARCH_TYPES = ["речь", "объект", "тип сцены", "упоминание", "глава"]
HEADERS = ["Дистрибутив", "Время", "Тип", "Что найдено", "Фрагмент"]

WORD_RE = re.compile(r"[a-zа-яё0-9][a-zа-яё0-9\-+]*", re.IGNORECASE)


def word_prefixes(query):
    """Ищу по основе слова, чтобы по «драйвер» находились и «драйвера», и «драйвером»."""
    words = WORD_RE.findall(query.lower().replace("ё", "е"))
    return [w if len(w) <= 4 else w[: max(4, len(w) - 2)] for w in words]


def build_items(index):
    items = []
    for v in index:
        for seg in v["transcript"].get("segments", []):
            items.append({"video": v, "type": "речь", "label": "", "start": seg["start"], "end": seg["end"],
                          "text": seg["text"], "search": seg["text"].lower().replace("ё", "е")})
        for e in v["events"]:
            kind = {"object": "объект", "scene": "тип сцены", "mention": "упоминание", "chapter": "глава"}.get(e["type"])
            if not kind:
                continue
            search = f"{e['label']} {e.get('class', '')} {EVENT_TYPES_RU[e['type']]}".lower()
            items.append({"video": v, "type": kind, "label": e["label"], "start": e["start"], "end": e["end"],
                          "text": e.get("text", ""), "search": search})
    return items


def search_items(items, query, distro="Все", types=None, limit=200):
    query = (query or "").strip()
    if not query:
        return []
    prefixes = word_prefixes(query)
    entities = {name for _, name in find_entities(query)}
    types = set(types or SEARCH_TYPES)
    hits = []
    for item in items:
        if item["type"] not in types or (distro != "Все" and item["video"]["distro"] != distro):
            continue
        if item["type"] == "упоминание" and entities:
            ok = item["label"] in entities
        else:
            ok = all(p in item["search"] for p in prefixes)
        if ok:
            hits.append(item)
    speech = {(h["video"]["id"], h["start"]) for h in hits if h["type"] == "речь"}
    hits = [h for h in hits if not (h["type"] == "упоминание" and (h["video"]["id"], h["start"]) in speech)]
    hits.sort(key=lambda h: (h["video"]["distro"], h["start"]))
    return hits[:limit]


def hits_table(hits):
    import pandas as pd
    rows = []
    for h in hits:
        label = h["label"] or "—"
        fragment = h["text"] or ""
        rows.append([h["video"]["distro"], f"{fmt_time(h['start'])}–{fmt_time(h['end'])}", h["type"],
                     label, fragment[:160]])
    return pd.DataFrame(rows, columns=HEADERS)


def player_html(video, start):
    start = int(start)
    if video["kind"] != "youtube":
        return f"<p>Локальный файл: data/{html.escape(video['id'])}/video.mp4, момент {fmt_time(start)}</p>"
    vid = html.escape(video["id"])
    return (
        f'<iframe width="100%" height="360" src="https://www.youtube.com/embed/{vid}?start={start}&autoplay=1" '
        f'title="{html.escape(video["distro"])}" frameborder="0" allow="autoplay; encrypted-media; picture-in-picture" '
        f'allowfullscreen></iframe>'
        f'<p><a href="{youtube_link(video["id"], start)}" target="_blank">Открыть на YouTube с {fmt_time(start)}</a></p>'
    )


def timeline_html(video):
    dur = float(video["probe"].get("duration") or video["info"].get("duration") or 1) or 1.0
    tracks = {"Главы": [], "Сцены": [], "Объекты": [], "Монтаж": [], "Звук": [], "Речь": []}
    for e in video["events"]:
        key = {"scene": "Сцены", "object": "Объекты", "scene_cut": "Монтаж", "loud": "Звук",
               "silence": "Звук", "mention": "Речь", "chapter": "Главы"}[e["type"]]
        tracks[key].append(e)
    rows = []
    for key, events in tracks.items():
        if not events:
            continue
        blocks = []
        for e in events:
            left = 100 * e["start"] / dur
            width = max(100 * (e["end"] - e["start"]) / dur, 0.3)
            color = SCENE_COLORS.get(e["label"], "#999") if e["type"] == "scene" else TRACK_COLORS[e["type"]]
            tip = f"{fmt_time(e['start'])}–{fmt_time(e['end'])} · {EVENT_TYPES_RU[e['type']]}: {e['label']}"
            href = youtube_link(video["id"], e["start"]) if video["kind"] == "youtube" else "#"
            blocks.append(
                f'<a href="{href}" target="_blank" title="{html.escape(tip)}" style="position:absolute;top:2px;'
                f'height:14px;left:{left:.3f}%;width:{width:.3f}%;min-width:2px;background:{color};border-radius:2px"></a>'
            )
        rows.append(
            f'<div style="display:flex;align-items:center;margin:3px 0"><span style="width:70px;font-size:12px;'
            f'opacity:.75">{key}</span><div style="position:relative;flex:1;height:18px;background:rgba(127,127,127,.15);'
            f'border-radius:3px">{"".join(blocks)}</div></div>'
        )
    legend = " ".join(
        f'<span style="display:inline-block;margin-right:10px;font-size:12px"><span style="display:inline-block;'
        f'width:10px;height:10px;background:{c};border-radius:2px;margin-right:3px"></span>{html.escape(n)}</span>'
        for n, c in SCENE_COLORS.items()
    )
    return (f'<div>{"".join(rows)}<div style="display:flex;justify-content:space-between;font-size:11px;opacity:.7;'
            f'margin-left:70px"><span>00:00</span><span>{fmt_time(dur / 2)}</span><span>{fmt_time(dur)}</span></div>'
            f'<div style="margin-top:6px">{legend}</div>'
            f'<p style="font-size:12px;opacity:.75">Наведите на блок, чтобы увидеть событие; клик открывает этот момент на YouTube.</p></div>')


def video_card(video, clusters):
    info, probe = video["info"], video["probe"]
    frames = max(len(video["frames"]), 1)
    objects = Counter()
    seen = {}
    for d in video["detections"]:
        if float(d["conf"]) >= YOLO_CONF:
            seen.setdefault(d["class"], set()).add(d["t"])
    for cls, ts in seen.items():
        objects[OBJECT_NAMES_RU.get(cls, cls)] = len(ts) / frames
    scenes = Counter(s["scene"] for s in video["scenes"] if float(s["score"]) >= SCENE_MIN_SCORE)
    scene_total = max(len(video["scenes"]), 1)
    words = len(video["transcript"].get("text", "").split())
    cluster = next((c for c in clusters if c["id"] == video["id"]), None)
    views = info.get("view_count") or 0
    likes = info.get("like_count") or 0
    comments = info.get("comment_count") or 0
    engagement = f"{100 * (likes + comments) / views:.2f}%" if views else "—"
    lines = [
        f"### {video['distro']}: {info.get('title', '')}",
        f"**Канал:** {info.get('uploader') or video.get('channel', '')} · **Длительность:** {fmt_time(probe.get('duration') or 0)}"
        f" · **{probe.get('width')}×{probe.get('height')}, {probe.get('fps')} fps, {probe.get('video_codec')}**",
        f"**Просмотры:** {views:,} · **Лайки:** {likes:,} · **Комментарии:** {comments:,} · **Engagement:** {engagement}".replace(",", " "),
        f"**Тема/кластер:** {('кластер ' + cluster['cluster'] + ', тема NMF ' + cluster['main_topic']) if cluster else '—'}",
        f"**Типы сцен:** {', '.join(f'{s} {n / scene_total:.0%}' for s, n in scenes.most_common(4)) or '—'}",
        f"**Объекты (доля кадров):** {', '.join(f'{o} {p:.0%}' for o, p in objects.most_common(5)) or '—'}",
        f"**Речь:** {words} слов · **Сущности:** "
        + (", ".join(f"{n} ({d['count']})" for n, d in list(video["entities"].items())[:8]) or "—"),
    ]
    if video["transcript"].get("empty"):
        lines.append("⚠️ Транскрипт пустой — речь не распознана.")
    return "\n\n".join(lines)


def main():
    import gradio as gr

    ensure_cwd()
    index = load_index()
    if not index:
        raise SystemExit("Нет результатов: сначала запустите pipeline.py")
    items = build_items(index)
    clusters = read_csv(RESULTS_DIR / "clusters.csv")
    by_distro = {v["distro"]: v for v in index}
    distros = ["Все"] + [v["distro"] for v in index]

    def on_search(query, distro, types):
        hits = search_items(items, query, distro, types)
        status = f"Найдено: {len(hits)}" + (" (показаны первые 200)" if len(hits) == 200 else "")
        state = [{"distro": h["video"]["distro"], "start": h["start"]} for h in hits]
        return hits_table(hits), state, status

    def on_select(state, evt: gr.SelectData):
        row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        if not state or row is None or row >= len(state):
            return gr.update(), gr.update(), gr.update()
        hit = state[row]
        video = by_distro[hit["distro"]]
        return player_html(video, hit["start"]), video_card(video, clusters), timeline_html(video)

    def on_video(distro):
        if distro == "Все":
            return "", "", ""
        video = by_distro[distro]
        return player_html(video, 0), video_card(video, clusters), timeline_html(video)

    with gr.Blocks(title="Поиск по видео о Linux") as demo:
        gr.Markdown("## Поиск по видео о дистрибутивах Linux\n"
                    "Ищите слово из речи (`драйвер`, `установка`), объект (`ноутбук`, `человек`), "
                    "тип сцены (`терминал`, `игра`) или дистрибутив (`убунту`, `arch`). "
                    "Клик по строке результата — переход к этому моменту ролика.")
        with gr.Row():
            query = gr.Textbox(label="Запрос", placeholder="например: драйвер nvidia", scale=4)
            distro = gr.Dropdown(distros, value="Все", label="Ролик", scale=2)
            button = gr.Button("Найти", variant="primary", scale=1)
        types = gr.CheckboxGroup(SEARCH_TYPES, value=SEARCH_TYPES, label="Где искать")
        status = gr.Markdown()
        table = gr.Dataframe(headers=HEADERS, datatype=["str"] * 5, interactive=False, wrap=True)
        state = gr.State([])
        with gr.Row():
            with gr.Column(scale=3):
                player = gr.HTML()
            with gr.Column(scale=2):
                card = gr.Markdown()
        timeline = gr.HTML()

        with gr.Accordion("Темы и статистика", open=False):
            topic_images = [p for p in ("topics_map.png", "topics_doc_topic.png", "scene_types.png",
                                        "objects_heatmap.png", "entities_heatmap.png") if (RESULTS_DIR / p).exists()]
            if topic_images:
                gr.Gallery([str(RESULTS_DIR / p) for p in topic_images], columns=2, height="auto")
            if clusters:
                gr.Dataframe([[c["distro"], c["cluster"], c["main_topic"]] for c in clusters],
                             headers=["Дистрибутив", "Кластер", "Главная тема NMF"], interactive=False)

        button.click(on_search, [query, distro, types], [table, state, status])
        query.submit(on_search, [query, distro, types], [table, state, status])
        table.select(on_select, [state], [player, card, timeline])
        distro.change(on_video, [distro], [player, card, timeline])

    demo.launch(inbrowser=True)


if __name__ == "__main__":
    main()
