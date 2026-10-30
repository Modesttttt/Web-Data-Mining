# обрабатываю ролики из videos.csv по этапам и сохраняю результат каждого этапа в data/<id>/

import argparse
import json
import shutil
import subprocess
import sys
import time
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common import (
    CLIP_MODEL, EMPTY_TRANSCRIPT_CHARS, FRAME_CHANGE_THR, FRAME_CHECK_FPS, FRAME_MAX_INTERVAL,
    FRAME_MIN_INTERVAL, FRAME_WIDTH, LOUD_MAD_K, LOUD_WINDOW, OBJECT_GAP, OBJECT_MIN_FRAMES,
    OBJECT_NAMES_RU, SCENE_CUT_MIN_GAP, SCENE_CUT_THR, SCENE_LABELS, SCENE_MIN_DURATION,
    SCENE_MIN_SCORE, SILENCE_DB, SILENCE_MIN_DURATION, WHISPER_MODEL, YOLO_CONF, YOLO_CONF_SAVE,
    YOLO_MODEL, clear_skip, ensure_cwd, load_videos, read_csv, read_json, record_skip, rel,
    video_dir, write_csv, write_json,
)
from entities import find_entities

STAGES = ["download", "probe", "decompose", "detect", "scenes", "audio", "transcribe", "events"]

INFO_FIELDS = [
    "id", "title", "uploader", "channel", "upload_date", "duration", "view_count", "like_count",
    "comment_count", "tags", "categories", "description", "chapters", "webpage_url",
    "width", "height", "fps", "language",
]


def torch_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def find_video_file(folder):
    for path in sorted(folder.glob("video.*")):
        if path.suffix not in {".part", ".ytdl", ".json"}:
            return path
    return None


def stage_download(video, args):
    folder = video_dir(video["id"])
    folder.mkdir(parents=True, exist_ok=True)
    if find_video_file(folder) and (folder / "info.json").exists() and not args.force:
        return

    if video["kind"] == "local":
        source = Path(video["url"])
        if not source.exists():
            raise FileNotFoundError(f"нет локального файла {source}")
        target = folder / f"video{source.suffix.lower()}"
        shutil.copy2(source, target)
        info = {"id": video["id"], "title": video.get("title") or source.stem,
                "uploader": video.get("channel", ""), "webpage_url": "", "description": ""}
    else:
        import yt_dlp

        opts = {
            "format": "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/bv*[height<=720]+ba/b",
            "merge_output_format": "mp4",
            "outtmpl": str(folder / "video.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "noprogress": False,
            "retries": 5,
        }
        if args.cookies_from_browser:
            opts["cookiesfrombrowser"] = (args.cookies_from_browser,)
        with yt_dlp.YoutubeDL(opts) as ydl:
            raw = ydl.sanitize_info(ydl.extract_info(video["url"], download=True))
        info = {key: raw.get(key) for key in INFO_FIELDS}

    info["distro"] = video["distro"]
    info["source"] = video["url"]
    write_json(folder / "info.json", info)
    if not find_video_file(folder):
        raise RuntimeError("видео не сохранилось (проверьте, что установлен ffmpeg)")


def fraction(value):
    try:
        num, den = str(value).split("/")
        return round(float(num) / float(den), 3) if float(den) else 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0


def stage_probe(video, args):
    folder = video_dir(video["id"])
    out = folder / "probe.json"
    if out.exists() and not args.force:
        return
    path = find_video_file(folder)
    raw = subprocess.check_output(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", str(path)]
    )
    data = json.loads(raw.decode("utf-8", errors="replace"))
    fmt = data.get("format", {})
    v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    a = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), {})
    duration = float(fmt.get("duration") or 0)
    size = int(fmt.get("size") or 0)
    summary = {
        "container": fmt.get("format_name"),
        "duration": round(duration, 2),
        "size_mb": round(size / 2 ** 20, 1),
        "bit_rate_kbps": round(int(fmt.get("bit_rate") or 0) / 1000),
        "video_codec": v.get("codec_name"),
        "profile": v.get("profile"),
        "width": v.get("width"),
        "height": v.get("height"),
        "fps": fraction(v.get("avg_frame_rate")),
        "pix_fmt": v.get("pix_fmt"),
        "video_bit_rate_kbps": round(int(v.get("bit_rate") or 0) / 1000),
        "audio_codec": a.get("codec_name"),
        "sample_rate": int(a.get("sample_rate") or 0),
        "channels": a.get("channels"),
        "has_audio": bool(a),
    }
    write_json(out, summary)


def hsv_hist(frame_bgr):
    import cv2
    small = cv2.resize(frame_bgr, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def hist_distance(h1, h2):
    import cv2
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA))


def save_jpg(path, image):
    import cv2
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError(f"не удалось сжать кадр {path}")
    Path(path).write_bytes(buf.tobytes())


def extract_frames(video, args):
    """Беру кадр, когда картинка заметно изменилась, но не реже раза в 2 секунды."""
    import cv2

    folder = video_dir(video["id"])
    frames_dir = folder / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)

    cap = cv2.VideoCapture(rel(find_video_file(folder)))
    if not cap.isOpened():
        raise RuntimeError("OpenCV не открыл видео")
    fps = cap.get(cv2.CAP_PROP_FPS) or read_json(folder / "probe.json", {}).get("fps") or 25.0
    check_step = max(1, int(round(fps / FRAME_CHECK_FPS)))

    frames, motion = [], []
    prev_hist, last_kept_hist, last_kept_t = None, None, -1e9
    idx = 0
    while True:
        if idx % check_step:
            if not cap.grab():
                break
            idx += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        idx += 1

        hist = hsv_hist(frame)
        motion.append({"t": round(t, 2), "dist": round(hist_distance(prev_hist, hist), 4) if prev_hist is not None else 0.0})
        prev_hist = hist

        change = hist_distance(last_kept_hist, hist) if last_kept_hist is not None else 1.0
        since = t - last_kept_t
        if since >= FRAME_MAX_INTERVAL or (change >= FRAME_CHANGE_THR and since >= FRAME_MIN_INTERVAL):
            h, w = frame.shape[:2]
            if w > FRAME_WIDTH:
                frame = cv2.resize(frame, (FRAME_WIDTH, int(h * FRAME_WIDTH / w)), interpolation=cv2.INTER_AREA)
            name = f"{t:08.2f}.jpg"
            save_jpg(frames_dir / name, frame)
            frames.append({"t": round(t, 2), "file": f"frames/{name}", "change": round(change, 4),
                           "reason": "interval" if since >= FRAME_MAX_INTERVAL else "change"})
            last_kept_hist, last_kept_t = hist, t
    cap.release()

    if not frames:
        raise RuntimeError("не извлечено ни одного кадра")
    write_csv(folder / "frames.csv", frames, ["t", "file", "change", "reason"])
    write_csv(folder / "motion.csv", motion, ["t", "dist"])
    duration = motion[-1]["t"] if motion else 0
    print(f"  кадров: {len(frames)} на {duration / 60:.1f} мин "
          f"({len(frames) / max(duration, 1):.2f} кадр/с, по изменению: "
          f"{sum(f['reason'] == 'change' for f in frames)})")


def extract_audio(video):
    folder = video_dir(video["id"])
    probe = read_json(folder / "probe.json", {})
    if probe and not probe.get("has_audio", True):
        print("  в видео нет аудиодорожки")
        return
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(find_video_file(folder)), "-vn",
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(folder / "audio.wav")],
        check=True,
    )


def stage_decompose(video, args):
    folder = video_dir(video["id"])
    if not (folder / "frames.csv").exists() or args.force:
        extract_frames(video, args)
    if not (folder / "audio.wav").exists() or args.force:
        extract_audio(video)

_models = {}


def load_frame(folder, file):
    import cv2
    data = np.fromfile(str(folder / file), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def stage_detect(video, args):
    folder = video_dir(video["id"])
    out = folder / "detections.csv"
    if out.exists() and not args.force:
        return
    if "yolo" not in _models:
        from ultralytics import YOLO
        _models["yolo"] = YOLO(args.yolo_model)
    model = _models["yolo"]
    device = 0 if torch_device() == "cuda" else "cpu"

    frames = read_csv(folder / "frames.csv")
    rows = []
    for start in range(0, len(frames), args.batch):
        batch = frames[start:start + args.batch]
        images = [load_frame(folder, f["file"]) for f in batch]
        results = model.predict(images, conf=YOLO_CONF_SAVE, imgsz=640, device=device, verbose=False)
        for frame, image, result in zip(batch, images, results):
            h, w = image.shape[:2]
            for cls, conf, box in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(),
                                      result.boxes.xyxy.tolist()):
                x1, y1, x2, y2 = box
                rows.append({
                    "t": frame["t"], "class": model.names[int(cls)], "conf": round(conf, 3),
                    "x1": round(x1 / w, 3), "y1": round(y1 / h, 3),
                    "x2": round(x2 / w, 3), "y2": round(y2 / h, 3),
                })
    write_csv(out, rows, ["t", "class", "conf", "x1", "y1", "x2", "y2"])
    strong = Counter(r["class"] for r in rows if r["conf"] >= YOLO_CONF)
    print(f"  детекций: {len(rows)} (conf ≥ {YOLO_CONF}: {sum(strong.values())}), "
          f"чаще всего: {', '.join(f'{c} {n}' for c, n in strong.most_common(4)) or 'нет'}")


def stage_scenes(video, args):
    folder = video_dir(video["id"])
    out = folder / "scenes.csv"
    if out.exists() and not args.force:
        return
    if "clip" not in _models:
        from transformers import pipeline
        device = 0 if torch_device() == "cuda" else -1
        _models["clip"] = pipeline("zero-shot-image-classification", model=args.clip_model, device=device)
    clf = _models["clip"]
    from PIL import Image

    labels_en = list(SCENE_LABELS)
    frames = read_csv(folder / "frames.csv")
    rows = []
    for start in range(0, len(frames), args.batch):
        batch = frames[start:start + args.batch]
        images = [Image.open(folder / f["file"]).convert("RGB") for f in batch]
        outputs = clf(images, candidate_labels=labels_en, hypothesis_template="{}", batch_size=len(images))
        if outputs and isinstance(outputs[0], dict):
            outputs = [outputs]
        for frame, scores in zip(batch, outputs):
            by_label = {SCENE_LABELS[s["label"]]: s["score"] for s in scores}
            best = max(by_label, key=by_label.get)
            row = {"t": frame["t"], "scene": best, "score": round(by_label[best], 3)}
            row.update({name: round(by_label.get(name, 0.0), 3) for name in SCENE_LABELS.values()})
            rows.append(row)
    write_csv(out, rows, ["t", "scene", "score"] + list(SCENE_LABELS.values()))
    share = Counter(r["scene"] for r in rows if r["score"] >= SCENE_MIN_SCORE)
    print(f"  сцены: {', '.join(f'{s} {n}' for s, n in share.most_common(4))}")


def read_wav(path):
    with wave.open(str(path), "rb") as wav:
        rate = wav.getframerate()
        data = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    return data.astype(np.float32) / 32768.0, rate


def stage_audio(video, args):
    folder = video_dir(video["id"])
    out = folder / "loudness.csv"
    if (out.exists() and not args.force) or not (folder / "audio.wav").exists():
        return
    samples, rate = read_wav(folder / "audio.wav")
    win = int(rate * LOUD_WINDOW)
    rows = []
    for i in range(0, len(samples) - win + 1, win):
        rms = float(np.sqrt(np.mean(samples[i:i + win] ** 2)))
        rows.append({"t": round(i / rate, 2), "db": round(20 * np.log10(max(rms, 1e-6)), 2)})
    write_csv(out, rows, ["t", "db"])


def stage_transcribe(video, args):
    folder = video_dir(video["id"])
    out = folder / "transcript.json"
    if out.exists() and not args.force:
        return
    if not (folder / "audio.wav").exists():
        write_json(out, {"language": None, "text": "", "segments": [], "empty": True,
                         "warning": "нет аудиодорожки"})
        record_skip(video["id"], "transcribe", "нет аудиодорожки — транскрипт пустой")
        return
    if "whisper" not in _models:
        import whisper
        _models["whisper"] = whisper.load_model(args.whisper_model, device=torch_device())
    model = _models["whisper"]
    samples, _ = read_wav(folder / "audio.wav")
    started = time.time()
    result = model.transcribe(samples, language="ru", fp16=torch_device() == "cuda",
                              condition_on_previous_text=False, verbose=None)
    segments = [{"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"].strip()}
                for s in result.get("segments", []) if s["text"].strip()]
    text = " ".join(s["text"] for s in segments)
    empty = len(text) < EMPTY_TRANSCRIPT_CHARS
    write_json(out, {"language": result.get("language"), "model": args.whisper_model,
                     "text": text, "segments": segments, "empty": empty})
    if empty:
        record_skip(video["id"], "transcribe", f"транскрипт пустой ({len(text)} символов)")
    print(f"  транскрипт: {len(segments)} сегментов, {len(text.split())} слов, "
          f"{time.time() - started:.0f} с")


def merge_times(times, gap):
    """Склеиваю отдельные моменты в интервалы, если соседние ближе gap."""
    intervals = []
    for t in sorted(times):
        if intervals and t - intervals[-1][1] <= gap:
            intervals[-1][1] = t
            intervals[-1][2] += 1
        else:
            intervals.append([t, t, 1])
    return intervals


def scene_cut_events(motion):
    events, last = [], -1e9
    for row in motion:
        t, dist = float(row["t"]), float(row["dist"])
        if dist >= SCENE_CUT_THR and t - last >= SCENE_CUT_MIN_GAP:
            events.append({"type": "scene_cut", "label": "смена сцены", "start": t, "end": t,
                           "score": round(dist, 3)})
            last = t
    return events


def object_events(detections):
    times = defaultdict(set)
    confs = defaultdict(list)
    for row in detections:
        conf = float(row["conf"])
        if conf >= YOLO_CONF:
            times[row["class"]].add(float(row["t"]))
            confs[row["class"]].append(conf)
    events = []
    for cls, ts in times.items():
        name = OBJECT_NAMES_RU.get(cls, cls)
        for start, end, n in merge_times(ts, OBJECT_GAP):
            if n >= OBJECT_MIN_FRAMES:
                events.append({"type": "object", "label": name, "class": cls, "start": start, "end": end,
                               "score": round(float(np.mean(confs[cls])), 3), "frames": n})
    return events


def scene_events(scenes):
    labels = [r["scene"] if float(r["score"]) >= SCENE_MIN_SCORE else "другое" for r in scenes]
    if len(labels) >= 3:
        labels = [Counter(labels[max(0, i - 1):i + 2]).most_common(1)[0][0] for i in range(len(labels))]
    events = []
    for row, label in zip(scenes, labels):
        t = float(row["t"])
        if events and events[-1]["label"] == label:
            events[-1]["end"] = t
            events[-1]["scores"].append(float(row[label]) if label in row else 0)
        else:
            events.append({"type": "scene", "label": label, "start": t, "end": t,
                           "scores": [float(row[label]) if label in row else 0]})
    result = []
    for e in events:
        if e["label"] != "другое" and e["end"] - e["start"] >= SCENE_MIN_DURATION:
            e["score"] = round(float(np.mean(e.pop("scores"))), 3)
            result.append(e)
    return result


def loudness_events(loudness):
    if not loudness:
        return [], {}
    t = np.array([float(r["t"]) for r in loudness])
    db = np.array([float(r["db"]) for r in loudness])
    voiced = db[db > SILENCE_DB]
    base = voiced if len(voiced) else db
    median = float(np.median(base))
    mad = max(float(np.median(np.abs(base - median))), 1.5)
    peak_thr = median + LOUD_MAD_K * mad

    events = []
    for start, end, n in merge_times(t[db >= peak_thr].tolist(), LOUD_WINDOW * 2):
        mask = (t >= start) & (t <= end)
        events.append({"type": "loud", "label": "пик громкости", "start": start, "end": end + LOUD_WINDOW,
                       "score": round(float(db[mask].max()), 1)})
    for start, end, n in merge_times(t[db < SILENCE_DB].tolist(), LOUD_WINDOW * 1.5):
        if end + LOUD_WINDOW - start >= SILENCE_MIN_DURATION:
            events.append({"type": "silence", "label": "тишина", "start": start, "end": end + LOUD_WINDOW,
                           "score": round(float(db[(t >= start) & (t <= end)].mean()), 1)})
    stats = {"median_db": round(median, 1), "mad_db": round(mad, 2), "peak_threshold_db": round(peak_thr, 1),
             "silence_share": round(float(np.mean(db < SILENCE_DB)), 3)}
    return events, stats


def mention_events(segments):
    events, counter, groups = [], Counter(), {}
    for seg in segments:
        found = find_entities(seg["text"])
        for group, name in dict.fromkeys(found):
            events.append({"type": "mention", "label": name, "group": group, "start": seg["start"],
                           "end": seg["end"], "score": 1.0, "text": seg["text"]})
        for group, name in found:
            counter[name] += 1
            groups[name] = group
    return events, {name: {"count": n, "group": groups[name]} for name, n in counter.most_common()}


def chapter_events(info):
    return [{"type": "chapter", "label": c.get("title", ""), "start": float(c.get("start_time", 0)),
             "end": float(c.get("end_time", 0)), "score": 1.0} for c in info.get("chapters") or []]


def stage_events(video, args):
    folder = video_dir(video["id"])
    info = read_json(folder / "info.json", {})
    transcript = read_json(folder / "transcript.json", {"segments": []})
    loud, loud_stats = loudness_events(read_csv(folder / "loudness.csv"))
    mentions, entities = mention_events(transcript.get("segments", []))
    events = (
        chapter_events(info)
        + scene_cut_events(read_csv(folder / "motion.csv"))
        + scene_events(read_csv(folder / "scenes.csv"))
        + object_events(read_csv(folder / "detections.csv"))
        + loud
        + mentions
    )
    for e in events:
        e["start"], e["end"] = round(float(e["start"]), 2), round(float(e["end"]), 2)
    events.sort(key=lambda e: (e["start"], e["type"]))
    write_json(folder / "events.json", events)
    write_json(folder / "entities.json", entities)
    write_json(folder / "audio_stats.json", loud_stats)
    counts = Counter(e["type"] for e in events)
    print(f"  событий: {len(events)} ({', '.join(f'{k} {v}' for k, v in counts.items())})")


RUNNERS = {
    "download": stage_download, "probe": stage_probe, "decompose": stage_decompose,
    "detect": stage_detect, "scenes": stage_scenes, "audio": stage_audio,
    "transcribe": stage_transcribe, "events": stage_events,
}
NEEDS_VIDEO = {"probe", "decompose"}
NEEDS_FRAMES = {"detect", "scenes"}


def main():
    parser = argparse.ArgumentParser(description="Пайплайн анализа видео (ЛР6)")
    parser.add_argument("--stage", choices=STAGES + ["all"], default="all")
    parser.add_argument("--only", nargs="*", help="ID роликов")
    parser.add_argument("--force", action="store_true", help="пересчитать готовые этапы")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--yolo-model", default=YOLO_MODEL)
    parser.add_argument("--clip-model", default=CLIP_MODEL)
    parser.add_argument("--whisper-model", default=WHISPER_MODEL)
    parser.add_argument("--cookies-from-browser", default=None,
                        help="firefox / edge / chrome — если YouTube просит войти")
    args = parser.parse_args()

    ensure_cwd()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    videos = load_videos()
    if args.only:
        videos = [v for v in videos if v["id"] in args.only]
    stages = STAGES if args.stage == "all" else [args.stage]
    print(f"Роликов: {len(videos)}, этапы: {', '.join(stages)}, устройство: {torch_device()}")

    for stage in stages:
        print(f"\n=== {stage} ===")
        for video in videos:
            folder = video_dir(video["id"])
            if stage in NEEDS_VIDEO and not find_video_file(folder):
                print(f"[{video['distro']}] нет видео, этап пропущен")
                continue
            if stage in NEEDS_FRAMES and not (folder / "frames.csv").exists():
                print(f"[{video['distro']}] нет кадров, этап пропущен")
                continue
            if stage != "download" and not (folder / "info.json").exists():
                continue
            print(f"[{video['distro']}] {video['id']}")
            clear_skip(video["id"], stage)
            try:
                RUNNERS[stage](video, args)
            except Exception as error:
                record_skip(video["id"], stage, f"{type(error).__name__}: {error}")

    skipped = read_json(Path("data/skipped.json"), [])
    if skipped:
        print(f"\nЕсть пропуски/предупреждения ({len(skipped)}), см. data/skipped.json")


if __name__ == "__main__":
    main()
