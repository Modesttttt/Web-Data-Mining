import csv
import json
import os
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
VIDEOS_CSV = BASE_DIR / "videos.csv"
SKIPPED_JSON = DATA_DIR / "skipped.json"

YOUTUBE_ID_RE = re.compile(r"(?:v=|youtu\.be/|shorts/|embed/)([\w-]{11})")

# Пороги анализа. Почему выбрал именно такие значения, расписал в README
FRAME_CHECK_FPS = 4          # 4 раза в секунду сравниваю кадр с последним сохранённым
FRAME_MIN_INTERVAL = 0.5     # беру не чаще 2 кадров в секунду
FRAME_MAX_INTERVAL = 2.0     # и не реже 1 кадра в 2 секунды
FRAME_CHANGE_THR = 0.20      # если картинка изменилась сильнее, сохраняю кадр раньше срока
FRAME_WIDTH = 640            # ужимаю кадры до 640 по ширине, YOLO всё равно работает на 640

SCENE_CUT_THR = 0.35         # выше этого расстояния между гистограммами считаю, что сцена сменилась
SCENE_CUT_MIN_GAP = 1.0      # склейки ближе секунды друг к другу считаю одной

YOLO_MODEL = "yolo26s.pt"
YOLO_CONF_SAVE = 0.25        # сохраняю всё от 0.25, чтобы потом менять порог без перезапуска YOLO
YOLO_CONF = 0.50             # в события и статистику беру только от 0.5
OBJECT_GAP = 3.0             # если объект пропал больше чем на 3 с, начинаю новый интервал
OBJECT_MIN_FRAMES = 2        # одиночное срабатывание за событие не считаю

CLIP_MODEL = "openai/clip-vit-base-patch32"
SCENE_MIN_SCORE = 0.40       # если CLIP уверен меньше, пишу сцену как «другое»
SCENE_MIN_DURATION = 4.0     # сцены короче 4 секунд не считаю событием

WHISPER_MODEL = "small"
EMPTY_TRANSCRIPT_CHARS = 50  # если текста меньше, считаю транскрипт пустым

LOUD_WINDOW = 0.5            # громкость считаю по окнам в полсекунды
LOUD_MAD_K = 3.0             # пик громкости - когда громче медианы на 3 MAD
SILENCE_DB = -45.0           # тишина - тише -45 dBFS
SILENCE_MIN_DURATION = 2.0

# Типы сцен для CLIP. Подписи пишу по-английски, так модель понимает их лучше
SCENE_LABELS = {
    "a Linux terminal window with lines of text commands": "терминал",
    "a computer desktop with application windows and a taskbar": "рабочий стол",
    "an operating system installation wizard": "установщик ОС",
    "a system settings window with options and switches": "настройки",
    "a web browser showing a website": "браузер",
    "a person talking to the camera": "человек в кадре",
    "a screenshot of a 3D video game": "игра",
    "a title screen with a big logo or text": "заставка",
}

# Русские названия классов COCO, которые попадаются в обзорах
OBJECT_NAMES_RU = {
    "person": "человек", "laptop": "ноутбук", "tv": "монитор", "keyboard": "клавиатура",
    "mouse": "мышь", "cell phone": "телефон", "cup": "кружка", "chair": "стул",
    "book": "книга", "bottle": "бутылка", "remote": "пульт", "clock": "часы",
    "potted plant": "растение", "couch": "диван", "bed": "кровать", "dining table": "стол",
    "car": "машина", "cat": "кошка", "dog": "собака", "tie": "галстук", "backpack": "рюкзак",
}

EVENT_TYPES_RU = {
    "scene_cut": "смена сцены",
    "scene": "тип сцены",
    "object": "объект",
    "loud": "пик громкости",
    "silence": "тишина",
    "mention": "упоминание",
    "chapter": "глава",
}


def ensure_cwd():
    """OpenCV на Windows не открывает пути с кириллицей, поэтому работаю из папки проекта."""
    os.chdir(BASE_DIR)


def rel(path):
    return os.path.relpath(path, BASE_DIR).replace("\\", "/")


def video_dir(video_id):
    return DATA_DIR / video_id


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def load_videos(path=VIDEOS_CSV):
    videos = []
    for row in read_csv(path):
        source = row["url"].strip()
        match = YOUTUBE_ID_RE.search(source)
        if match:
            video_id, kind = match.group(1), "youtube"
        else:
            video_id, kind = re.sub(r"[^\w-]", "_", Path(source).stem), "local"
        videos.append({**row, "url": source, "id": video_id, "kind": kind})
    return videos


def fmt_time(seconds):
    seconds = int(round(float(seconds)))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def youtube_link(video_id, seconds=0):
    return f"https://www.youtube.com/watch?v={video_id}&t={int(seconds)}s"


def record_skip(video_id, stage, reason):
    skipped = read_json(SKIPPED_JSON, [])
    skipped = [s for s in skipped if not (s["id"] == video_id and s["stage"] == stage)]
    skipped.append({"id": video_id, "stage": stage, "reason": str(reason)[:500]})
    write_json(SKIPPED_JSON, skipped)
    print(f"  [пропуск] {video_id} на этапе {stage}: {reason}")


def clear_skip(video_id, stage):
    skipped = read_json(SKIPPED_JSON, [])
    kept = [s for s in skipped if not (s["id"] == video_id and s["stage"] == stage)]
    if len(kept) != len(skipped):
        write_json(SKIPPED_JSON, kept)


def load_index():
    """Собираю все результаты пайплайна по всем роликам, они нужны отчёту и интерфейсу."""
    index = []
    for video in load_videos():
        folder = video_dir(video["id"])
        info = read_json(folder / "info.json")
        if not info:
            continue
        index.append({
            **video,
            "info": info,
            "probe": read_json(folder / "probe.json", {}),
            "transcript": read_json(folder / "transcript.json", {"segments": [], "text": ""}),
            "events": read_json(folder / "events.json", []),
            "frames": read_csv(folder / "frames.csv"),
            "detections": read_csv(folder / "detections.csv"),
            "scenes": read_csv(folder / "scenes.csv"),
            "loudness": read_csv(folder / "loudness.csv"),
            "motion": read_csv(folder / "motion.csv"),
            "entities": read_json(folder / "entities.json", {}),
        })
    return index
