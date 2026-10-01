import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Verify Lab 4 spider output")
    parser.add_argument("file")
    parser.add_argument("--min", type=int, default=100)
    args = parser.parse_args()

    path = Path(args.file)
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as f:
            rows = json.load(f)
    elif path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        raise SystemExit("Поддерживаются только .json и .csv")

    if not isinstance(rows, list):
        raise SystemExit("Ожидался массив записей")

    urls = [row.get("url") for row in rows]
    titles = [row.get("title") for row in rows]
    missing = sum(not x for x in urls) + sum(not x for x in titles)
    duplicates = len(urls) - len(set(urls))

    print(f"Файл: {path}")
    print(f"Записей: {len(rows)}")
    print(f"Дубликатов URL: {duplicates}")
    print(f"Пустых обязательных полей: {missing}")

    assert len(rows) >= args.min, f"Нужно минимум {args.min} записей"
    assert duplicates == 0, "Найдены дубликаты URL"
    assert missing == 0, "Найдены пустые title/url"
    print("Проверка: OK")


if __name__ == "__main__":
    main()
