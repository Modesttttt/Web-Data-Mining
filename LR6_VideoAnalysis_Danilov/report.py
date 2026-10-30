# строю графики, таблицы и шкалу событий по результатам pipeline.py

import html
import random
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (
    EVENT_TYPES_RU, OBJECT_NAMES_RU, RESULTS_DIR, SCENE_CUT_THR, SCENE_LABELS, SCENE_MIN_SCORE,
    SILENCE_DB, YOLO_CONF, ensure_cwd, fmt_time, load_index, read_json, video_dir, write_csv,
    youtube_link,
)
from entities import entity_group

SCENE_COLORS = {
    "терминал": "#2f3b45", "рабочий стол": "#3b6ea8", "установщик ОС": "#e08a2c",
    "настройки": "#8a6bbf", "браузер": "#3a9d8f", "человек в кадре": "#d1495b",
    "игра": "#5bb04a", "заставка": "#c9a227",
}
TRACK_COLORS = {"object": "#7a8b99", "scene_cut": "#222222", "loud": "#e4572e",
                "silence": "#9db4c0", "mention": "#6a4c93", "chapter": "#c9a227"}

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})


def duration_of(video):
    return float(video["probe"].get("duration") or video["info"].get("duration") or 0) or 1.0


def short(name, n=14):
    return name if len(name) <= n else name[: n - 1] + "…"


def summary_rows(index):
    rows = []
    for v in index:
        info, probe = v["info"], v["probe"]
        views = info.get("view_count") or 0
        likes = info.get("like_count") or 0
        comments = info.get("comment_count") or 0
        rows.append({
            "distro": v["distro"], "id": v["id"], "title": info.get("title", ""),
            "channel": info.get("uploader") or info.get("channel") or v.get("channel", ""),
            "upload_date": info.get("upload_date") or "", "duration_min": round(duration_of(v) / 60, 1),
            "views": views, "likes": likes, "comments": comments,
            "engagement_pct": round(100 * (likes + comments) / views, 2) if views else "",
            "resolution": f"{probe.get('width')}x{probe.get('height')}", "fps": probe.get("fps"),
            "video_codec": probe.get("video_codec"), "audio_codec": probe.get("audio_codec"),
            "bitrate_kbps": probe.get("bit_rate_kbps"), "size_mb": probe.get("size_mb"),
        })
    return rows


def stats_rows(index):
    rows = []
    for v in index:
        dur = duration_of(v)
        minutes = dur / 60
        segs = v["transcript"].get("segments", [])
        words = sum(len(s["text"].split()) for s in segs)
        speech = sum(s["end"] - s["start"] for s in segs)
        strong = [d for d in v["detections"] if float(d["conf"]) >= YOLO_CONF]
        frames_total = max(len(v["frames"]), 1)
        objects = Counter()
        for cls, ts in group_times(strong).items():
            objects[cls] = len(ts) / frames_total
        scenes = Counter(s["scene"] for s in v["scenes"] if float(s["score"]) >= SCENE_MIN_SCORE)
        cuts = sum(e["type"] == "scene_cut" for e in v["events"])
        audio = read_json(video_dir(v["id"]) / "audio_stats.json", {})
        rows.append({
            "distro": v["distro"], "id": v["id"], "duration_min": round(minutes, 1),
            "frames": len(v["frames"]), "frames_per_min": round(len(v["frames"]) / minutes, 1),
            "scene_cuts": cuts, "cuts_per_min": round(cuts / minutes, 2),
            "top_objects": ", ".join(f"{OBJECT_NAMES_RU.get(c, c)} {p:.0%}" for c, p in objects.most_common(3)),
            "main_scene": scenes.most_common(1)[0][0] if scenes else "",
            "main_scene_share": round(scenes.most_common(1)[0][1] / max(len(v["scenes"]), 1), 2) if scenes else "",
            "words": words, "words_per_min": round(words / minutes), "speech_share": round(min(speech / dur, 1), 2),
            "transcript_empty": v["transcript"].get("empty", not segs),
            "silence_share": audio.get("silence_share", ""), "loud_peaks": sum(e["type"] == "loud" for e in v["events"]),
            "top_entities": ", ".join(list(v["entities"])[:5]),
            "events": len(v["events"]),
        })
    return rows


def group_times(detections):
    times = defaultdict(set)
    for d in detections:
        times[d["class"]].add(d["t"])
    return times


def plot_engagement(rows, path):
    rows = [r for r in rows if r["views"]]
    if not rows:
        return
    names = [r["distro"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    ax1.barh(names, [r["views"] / 1000 for r in rows], color="#3b6ea8")
    ax1.set_xlabel("Просмотры, тыс.")
    ax1.invert_yaxis()
    ax1.set_title("Просмотры")
    ax2.barh(names, [r["engagement_pct"] or 0 for r in rows], color="#e08a2c")
    ax2.set_xlabel("(лайки + комментарии) / просмотры, %")
    ax2.invert_yaxis()
    ax2.set_title("Вовлечённость (engagement rate)")
    for ax, key in ((ax1, "views"), (ax2, "engagement_pct")):
        for i, r in enumerate(rows):
            value = r[key] / 1000 if key == "views" else r[key] or 0
            ax.text(value, i, f" {value:.1f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_timeline(video, path):
    dur = duration_of(video)
    events = video["events"]
    objects = Counter(e["label"] for e in events if e["type"] == "object")
    top_objects = [name for name, _ in objects.most_common(4)]
    tracks = ["Сцена"] + [f"Объект: {o}" for o in top_objects] + ["Смена сцены", "Громкость", "Упоминания"]

    fig = plt.figure(figsize=(14, 1.2 + 0.42 * len(tracks) + 1.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.42 * len(tracks), 1.4], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    axl = fig.add_subplot(gs[1], sharex=ax)
    y = {name: i for i, name in enumerate(tracks)}

    for e in events:
        if e["type"] == "scene":
            ax.barh(y["Сцена"], max(e["end"] - e["start"], 1), left=e["start"], height=0.7,
                    color=SCENE_COLORS.get(e["label"], "#999"))
        elif e["type"] == "object" and e["label"] in top_objects:
            ax.barh(y[f"Объект: {e['label']}"], max(e["end"] - e["start"], 1), left=e["start"],
                    height=0.5, color=TRACK_COLORS["object"])
        elif e["type"] == "scene_cut":
            ax.vlines(e["start"], y["Смена сцены"] - 0.3, y["Смена сцены"] + 0.3, color="#222", lw=0.8)
        elif e["type"] == "mention":
            ax.vlines(e["start"], y["Упоминания"] - 0.3, y["Упоминания"] + 0.3, color=TRACK_COLORS["mention"], lw=1.2)
        elif e["type"] in ("loud", "silence"):
            ax.barh(y["Громкость"], max(e["end"] - e["start"], 0.5), left=e["start"], height=0.5,
                    color=TRACK_COLORS[e["type"]])

    mentions = Counter(e["label"] for e in events if e["type"] == "mention")
    first_seen = {}
    for e in events:
        if e["type"] == "mention" and e["label"] in dict(mentions.most_common(5)) and e["label"] not in first_seen:
            first_seen[e["label"]] = e["start"]
    for i, (label, t) in enumerate(sorted(first_seen.items(), key=lambda x: x[1])):
        ax.text(t, y["Упоминания"] + 0.42 - 0.28 * (i % 3), f" {label}", fontsize=7, color=TRACK_COLORS["mention"])

    ax.set_yticks(range(len(tracks)))
    ax.set_yticklabels(tracks, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, dur)
    ax.tick_params(labelbottom=False)
    ax.set_title(f"{video['distro']}: {short(video['info'].get('title', ''), 80)}", fontsize=11, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SCENE_COLORS.values()]
    ax.legend(handles, SCENE_COLORS.keys(), ncol=8, fontsize=7, loc="upper center",
              bbox_to_anchor=(0.5, -0.02 - 1.4 / (0.42 * len(tracks) + 1.4)), frameon=False)

    if video["loudness"]:
        t = np.array([float(r["t"]) for r in video["loudness"]])
        db = np.array([float(r["db"]) for r in video["loudness"]])
        axl.plot(t, np.clip(db, -60, 0), color="#555", lw=0.6)
        audio = read_json(video_dir(video["id"]) / "audio_stats.json", {})
        if audio.get("peak_threshold_db") is not None:
            axl.axhline(audio["peak_threshold_db"], color=TRACK_COLORS["loud"], ls="--", lw=0.8, label="порог пика")
        axl.axhline(SILENCE_DB, color=TRACK_COLORS["silence"], ls="--", lw=0.8, label="порог тишины")
        axl.legend(fontsize=7, loc="lower right")
    axl.set_ylabel("dBFS", fontsize=8)
    ticks = np.arange(0, dur + 1, 60 if dur > 300 else 15)
    axl.set_xticks(ticks)
    axl.set_xticklabels([fmt_time(x) for x in ticks], fontsize=8)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_scene_shares(index, path):
    labels = list(SCENE_COLORS)
    fig, ax = plt.subplots(figsize=(12, 0.45 * len(index) + 1.6))
    left = np.zeros(len(index))
    for label in labels + ["другое"]:
        shares = []
        for v in index:
            total = max(len(v["scenes"]), 1)
            if label == "другое":
                shares.append(sum(float(s["score"]) < SCENE_MIN_SCORE for s in v["scenes"]) / total)
            else:
                shares.append(sum(s["scene"] == label and float(s["score"]) >= SCENE_MIN_SCORE
                                  for s in v["scenes"]) / total)
        ax.barh([v["distro"] for v in index], shares, left=left, color=SCENE_COLORS.get(label, "#ddd"), label=label)
        left += np.array(shares)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Доля кадров")
    ax.set_title("Что показывают в роликах: тип сцены по кадрам (CLIP)")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def heatmap(matrix, rows, cols, title, path, fmt="{:.0%}", cmap="Blues", vmax=None):
    fig, ax = plt.subplots(figsize=(max(7, 0.75 * len(cols) + 3), 0.45 * len(rows) + 1.8))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=vmax or max(matrix.max(), 1e-9))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=9)
    for i in range(len(rows)):
        for j in range(len(cols)):
            value = matrix[i, j]
            if value:
                ax.text(j, i, fmt.format(value), ha="center", va="center", fontsize=7,
                        color="white" if value > 0.6 * im.get_clim()[1] else "black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_objects(index, path):
    totals = Counter()
    shares = []
    for v in index:
        frames = max(len(v["frames"]), 1)
        strong = [d for d in v["detections"] if float(d["conf"]) >= YOLO_CONF]
        share = {cls: len(ts) / frames for cls, ts in group_times(strong).items()}
        shares.append(share)
        totals.update(share)
    classes = [c for c, _ in totals.most_common(10)]
    if not classes:
        return
    matrix = np.array([[s.get(c, 0) for c in classes] for s in shares])
    heatmap(matrix, [v["distro"] for v in index], [OBJECT_NAMES_RU.get(c, c) for c in classes],
            f"Доля кадров, где YOLO нашла объект (conf ≥ {YOLO_CONF})", path)


def plot_entities(index, path_heat, path_top):
    totals = Counter()
    for v in index:
        totals.update({k: d["count"] for k, d in v["entities"].items()})
    names = [n for n, _ in totals.most_common(18)]
    if not names:
        return
    matrix = np.array([[v["entities"].get(n, {}).get("count", 0) for n in names] for v in index], dtype=float)
    heatmap(matrix, [v["distro"] for v in index], names, "Сколько раз сущность упомянута в речи (Whisper)",
            path_heat, fmt="{:.0f}", cmap="Purples")

    groups = sorted({entity_group(n) for n in names})
    palette = dict(zip(groups, ["#3b6ea8", "#e08a2c", "#5bb04a", "#d1495b", "#8a6bbf"]))
    fig, ax = plt.subplots(figsize=(10, 0.32 * len(names) + 1.5))
    ax.barh(names, [totals[n] for n in names], color=[palette[entity_group(n)] for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("Упоминаний во всех роликах")
    ax.set_title("Самые упоминаемые сущности")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in palette.values()], palette.keys(), fontsize=8)
    fig.tight_layout()
    fig.savefig(path_top)
    plt.close(fig)


def plot_speech(stats, path):
    names = [s["distro"] for s in stats]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 0.45 * len(names) + 1.8))
    ax1.barh(names, [s["words_per_min"] for s in stats], color="#6a4c93")
    ax1.set_xlabel("Слов в минуту")
    ax1.set_title("Темп речи")
    ax2.barh(names, [s["cuts_per_min"] for s in stats], color="#2f3b45")
    ax2.set_xlabel("Смен сцены в минуту")
    ax2.set_title("Динамика монтажа")
    for ax in (ax1, ax2):
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_thresholds(index, path):
    dist = np.array([float(m["dist"]) for v in index for m in v["motion"]])
    conf = np.array([float(d["conf"]) for v in index for d in v["detections"]])
    score = np.array([float(s["score"]) for v in index for s in v["scenes"]])
    db = np.array([float(r["db"]) for v in index for r in v["loudness"]])

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    panels = [
        (axes[0, 0], dist, SCENE_CUT_THR, "Разница соседних кадров (Бхаттачарья)", "смена сцены", True),
        (axes[0, 1], conf, YOLO_CONF, "Уверенность YOLO", "порог объекта", False),
        (axes[1, 0], score, SCENE_MIN_SCORE, "Уверенность CLIP в типе сцены", "порог сцены", False),
        (axes[1, 1], np.clip(db, -80, 0), SILENCE_DB, "Громкость окна 0,5 с, dBFS", "тишина", False),
    ]
    for ax, data, thr, title, label, log in panels:
        if len(data):
            ax.hist(data, bins=60, color="#7a8b99", log=log)
            share = np.mean(data >= thr) if label != "тишина" else np.mean(data < thr)
            ax.axvline(thr, color="#d1495b", ls="--", label=f"{label}: {thr} ({share:.1%} значений)")
            ax.legend(fontsize=8)
        ax.set_title(title, fontsize=10)
    fig.suptitle("Распределения признаков и выбранные пороги", fontsize=13)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def timeline_html(index, path):
    blocks = []
    legend = "".join(
        f'<span class="lg"><i style="background:{c}"></i>{html.escape(n)}</span>' for n, c in SCENE_COLORS.items()
    ) + "".join(
        f'<span class="lg"><i style="background:{TRACK_COLORS[k]}"></i>{EVENT_TYPES_RU[k]}</span>'
        for k in ("object", "scene_cut", "loud", "silence", "mention", "chapter")
    )
    for v in index:
        dur = duration_of(v)
        youtube = v["kind"] == "youtube"
        tracks = defaultdict(list)
        for e in v["events"]:
            key = {"scene": "Сцены", "object": "Объекты", "scene_cut": "Монтаж", "loud": "Звук",
                   "silence": "Звук", "mention": "Речь", "chapter": "Главы"}[e["type"]]
            tracks[key].append(e)
        rows = []
        for key in ("Главы", "Сцены", "Объекты", "Монтаж", "Звук", "Речь"):
            items = []
            for e in tracks.get(key, []):
                left = 100 * e["start"] / dur
                width = max(100 * (e["end"] - e["start"]) / dur, 0.25)
                color = SCENE_COLORS.get(e["label"], "#999") if e["type"] == "scene" else TRACK_COLORS[e["type"]]
                tip = f"{fmt_time(e['start'])}–{fmt_time(e['end'])} · {EVENT_TYPES_RU[e['type']]}: {e['label']}"
                if e.get("text"):
                    tip += f" · «{e['text'][:120]}»"
                href = youtube_link(v["id"], e["start"]) if youtube else "#"
                items.append(
                    f'<a class="ev" href="{href}" target="_blank" title="{html.escape(tip)}" '
                    f'style="left:{left:.3f}%;width:{width:.3f}%;background:{color}"></a>'
                )
            rows.append(f'<div class="row"><span class="name">{key}</span><div class="track">{"".join(items)}</div></div>')
        marks = "".join(
            f'<span class="tick" style="left:{100 * t / dur:.2f}%">{fmt_time(t)}</span>'
            for t in np.arange(0, dur, 60 if dur > 300 else 15)
        )
        link = youtube_link(v["id"]) if youtube else ""
        title = html.escape(v["info"].get("title", ""))
        head = f'<a href="{link}" target="_blank">{title}</a>' if link else title
        blocks.append(
            f'<section><h2>{html.escape(v["distro"])} <small>{head} · {fmt_time(dur)}</small></h2>'
            f'{"".join(rows)}<div class="row"><span class="name"></span><div class="axis">{marks}</div></div></section>'
        )
    page = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Шкала событий видео</title>
<style>
:root {{ --bg:#fafafa; --fg:#1d232a; --muted:#6b7680; --track:#e9edf0; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#15191d; --fg:#e8ecef; --muted:#9aa5ae; --track:#252c33; }} }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.4 system-ui, sans-serif; margin:0; padding:16px; }}
h1 {{ font-size:20px; margin:0 0 4px }} p {{ color:var(--muted); margin:0 0 12px }}
section {{ margin:18px 0 26px }} h2 {{ font-size:16px; margin:0 0 6px }} h2 small {{ font-weight:400; color:var(--muted) }}
h2 a {{ color:inherit }}
.row {{ display:flex; align-items:center; margin:3px 0 }}
.name {{ width:72px; flex:none; font-size:12px; color:var(--muted) }}
.track {{ position:relative; flex:1; height:16px; background:var(--track); border-radius:3px }}
.ev {{ position:absolute; top:2px; height:12px; border-radius:2px; min-width:2px; opacity:.9 }}
.ev:hover {{ outline:2px solid var(--fg); opacity:1; z-index:2 }}
.axis {{ position:relative; flex:1; height:16px }}
.tick {{ position:absolute; font-size:10px; color:var(--muted); transform:translateX(-50%) }}
.lg {{ display:inline-flex; align-items:center; margin:0 12px 4px 0; font-size:12px }}
.lg i {{ width:12px; height:12px; border-radius:2px; margin-right:4px; display:inline-block }}
</style></head><body>
<h1>Шкала событий: {len(index)} роликов о дистрибутивах Linux</h1>
<p>Наведите на событие, чтобы увидеть подробности; клик открывает ролик на YouTube с этой секунды.</p>
<div>{legend}</div>
{"".join(blocks)}
</body></html>"""
    path.write_text(page, encoding="utf-8")


def verification(index, folder, n_frames=24, n_segments=15, seed=42):
    import cv2

    folder.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    pool = [(v, f) for v in index for f in v["frames"]]
    sample = rng.sample(pool, min(n_frames, len(pool)))

    tiles, rows = [], []
    for v, f in sample:
        path = video_dir(v["id"]) / f["file"]
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = image.shape[:2]
        objects = []
        for d in v["detections"]:
            if d["t"] == f["t"] and float(d["conf"]) >= YOLO_CONF:
                x1, y1, x2, y2 = (float(d[k]) for k in ("x1", "y1", "x2", "y2"))
                cv2.rectangle(image, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (40, 200, 255), 2)
                objects.append(f"{d['class']} {float(d['conf']):.2f}")
        scene = next((s for s in v["scenes"] if s["t"] == f["t"]), None)
        rows.append({
            "n": len(rows) + 1, "distro": v["distro"], "t": fmt_time(f["t"]),
            "link": youtube_link(v["id"], float(f["t"])) if v["kind"] == "youtube" else "",
            "yolo": ", ".join(objects) or "—",
            "scene": f"{scene['scene']} {float(scene['score']):.2f}" if scene else "—",
            "yolo_ok": "", "scene_ok": "", "comment": "",
        })
        tiles.append((image, len(rows), v["distro"], f["t"], scene["scene"] if scene else "—", objects))

    cols = 4
    fig, axes = plt.subplots(int(np.ceil(len(tiles) / cols)), cols, figsize=(16, 2.6 * np.ceil(len(tiles) / cols)))
    for ax, (image, n, distro, t, scene, objects) in zip(np.atleast_1d(axes).flatten(), tiles):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(f"#{n} {distro} {fmt_time(t)} | {scene}\n{', '.join(o.split()[0] for o in objects) or 'нет объектов'}",
                     fontsize=8)
        ax.axis("off")
    for ax in np.atleast_1d(axes).flatten()[len(tiles):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(folder / "frames_check.jpg", dpi=90)
    plt.close(fig)
    write_csv(folder / "frames_check.csv", rows, list(rows[0]) if rows else ["n"])

    seg_pool = [(v, s) for v in index for s in v["transcript"].get("segments", [])]
    seg_rows = []
    for v, s in rng.sample(seg_pool, min(n_segments, len(seg_pool))):
        seg_rows.append({
            "distro": v["distro"], "start": fmt_time(s["start"]), "end": fmt_time(s["end"]),
            "link": youtube_link(v["id"], s["start"]) if v["kind"] == "youtube" else "",
            "whisper_text": s["text"], "errors": "", "comment": "",
        })
    if seg_rows:
        write_csv(folder / "transcript_check.csv", seg_rows, list(seg_rows[0]))


def main():
    ensure_cwd()
    index = load_index()
    if not index:
        raise SystemExit("Нет результатов: сначала запустите pipeline.py")
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "timelines").mkdir(exist_ok=True)

    summary = summary_rows(index)
    write_csv(RESULTS_DIR / "videos_summary.csv", summary, list(summary[0]))
    stats = stats_rows(index)
    write_csv(RESULTS_DIR / "video_stats.csv", stats, list(stats[0]))

    entity_rows = [{"distro": v["distro"], "entity": name, "group": d["group"], "count": d["count"]}
                   for v in index for name, d in v["entities"].items()]
    write_csv(RESULTS_DIR / "entities.csv", entity_rows, ["distro", "entity", "group", "count"])

    plot_engagement(summary, RESULTS_DIR / "engagement.png")
    plot_scene_shares(index, RESULTS_DIR / "scene_types.png")
    plot_objects(index, RESULTS_DIR / "objects_heatmap.png")
    plot_entities(index, RESULTS_DIR / "entities_heatmap.png", RESULTS_DIR / "entities_top.png")
    plot_speech(stats, RESULTS_DIR / "speech_and_cuts.png")
    plot_thresholds(index, RESULTS_DIR / "thresholds.png")
    for v in index:
        plot_timeline(v, RESULTS_DIR / "timelines" / f"{v['id']}.png")
    timeline_html(index, RESULTS_DIR / "timeline.html")
    verification(index, RESULTS_DIR / "verification")

    print(f"Роликов в отчёте: {len(index)}")
    print(f"{'дистрибутив':<14}{'мин':>6}{'кадров':>8}{'смен/мин':>10}{'слов/мин':>10}  главная сцена; объекты; сущности")
    for s in stats:
        print(f"{s['distro']:<14}{s['duration_min']:>6}{s['frames']:>8}{s['cuts_per_min']:>10}{s['words_per_min']:>10}"
              f"  {s['main_scene']}; {s['top_objects'] or '—'}; {s['top_entities'] or '—'}")
    empty = [s["distro"] for s in stats if s["transcript_empty"]]
    if empty:
        print(f"ВНИМАНИЕ: пустой транскрипт у {', '.join(empty)}")
    print("Готово: results/ (timeline.html — интерактивная шкала, verification/ — выборка для ручной проверки)")


if __name__ == "__main__":
    main()
