# загрузка MovieLens, разведочный анализ и разбиение на train/test

import io
import json
import re
import ssl
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DATASETS = {
    "movielens": ("ml-latest-small", "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"),
}

RELEVANT_RATING = 4.0
TEST_SHARE = 0.2

YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


def ssl_contexts():
    """Сначала проверяю сертификат через хранилище Windows (truststore), потом обычным способом и через certifi."""
    try:
        import truststore
        yield truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        pass
    yield ssl.create_default_context()
    try:
        import certifi
        yield ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass


def fetch(url):
    last_error = None
    for context in ssl_contexts():
        try:
            with urllib.request.urlopen(url, timeout=120, context=context) as response:
                return response.read()
        except (urllib.error.URLError, ssl.SSLError) as error:
            last_error = error
    raise SystemExit(f"Не получилось скачать {url}: {last_error}\n"
                     f"Скачайте архив вручную в браузере и положите его в {DATA_DIR}")


def download(name):
    folder_name, url = DATASETS[name]
    folder = DATA_DIR / folder_name
    if (folder / "ratings.csv").exists():
        return folder
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    local_zip = DATA_DIR / f"{folder_name}.zip"
    if local_zip.exists():
        print(f"Распаковываю {local_zip}")
        content = local_zip.read_bytes()
    else:
        print(f"Скачиваю {url}")
        content = fetch(url)
        local_zip.write_bytes(content)
    zipfile.ZipFile(io.BytesIO(content)).extractall(DATA_DIR)
    return folder


def load(name="movielens"):
    folder = download(name)
    ratings = pd.read_csv(folder / "ratings.csv")
    movies = pd.read_csv(folder / "movies.csv")
    tags = pd.read_csv(folder / "tags.csv") if (folder / "tags.csv").exists() else pd.DataFrame(
        columns=["userId", "movieId", "tag", "timestamp"])
    ratings = ratings.rename(columns={"userId": "user", "movieId": "item"})
    movies = movies.rename(columns={"movieId": "item"})
    tags = tags.rename(columns={"userId": "user", "movieId": "item"})
    movies["year"] = movies["title"].str.extract(YEAR_RE, expand=False).astype(float)
    movies["genre_list"] = movies["genres"].apply(
        lambda g: [] if g == "(no genres listed)" else g.split("|"))
    return ratings, movies, tags


def temporal_split(ratings, test_share=TEST_SHARE):
    """Для каждого пользователя в тест уходят его последние по времени 20% оценок."""
    ratings = ratings.sort_values(["user", "timestamp", "item"]).copy()
    rank = ratings.groupby("user").cumcount()
    size = ratings.groupby("user")["item"].transform("size")
    n_test = np.maximum(1, np.floor(size * test_share)).astype(int)
    is_test = rank >= size - n_test
    return ratings[~is_test].reset_index(drop=True), ratings[is_test].reset_index(drop=True)


def describe(ratings, movies, tags, train, test):
    n_users = ratings["user"].nunique()
    n_items_rated = ratings["item"].nunique()
    density = len(ratings) / (n_users * n_items_rated)
    per_user = ratings.groupby("user").size()
    per_item = ratings.groupby("item").size()
    top_items = per_item.sort_values(ascending=False)
    share_top10 = top_items.head(int(len(top_items) * 0.1)).sum() / len(ratings)
    train_items = set(train["item"])
    cold_test = test[~test["item"].isin(train_items)]
    relevant = test[test["rating"] >= RELEVANT_RATING]
    tagged = tags["item"].nunique()
    return {
        "ratings": int(len(ratings)),
        "users": int(n_users),
        "items_in_catalog": int(len(movies)),
        "items_rated": int(n_items_rated),
        "tags": int(len(tags)),
        "items_with_tags": int(tagged),
        "density": round(float(density), 5),
        "sparsity": round(float(1 - density), 5),
        "ratings_per_user_min": int(per_user.min()),
        "ratings_per_user_median": float(per_user.median()),
        "ratings_per_user_max": int(per_user.max()),
        "ratings_per_item_median": float(per_item.median()),
        "items_with_1_rating": int((per_item == 1).sum()),
        "share_ratings_top10pct_items": round(float(share_top10), 3),
        "mean_rating": round(float(ratings["rating"].mean()), 3),
        "share_relevant": round(float((ratings["rating"] >= RELEVANT_RATING).mean()), 3),
        "train_ratings": int(len(train)),
        "test_ratings": int(len(test)),
        "test_relevant": int(len(relevant)),
        "test_users_with_relevant": int(relevant["user"].nunique()),
        "test_ratings_cold_items": int(len(cold_test)),
        "test_relevant_cold_items": int((cold_test["rating"] >= RELEVANT_RATING).sum()),
        "period": [str(pd.to_datetime(ratings["timestamp"].min(), unit="s").date()),
                   str(pd.to_datetime(ratings["timestamp"].max(), unit="s").date())],
    }


def plot_eda(ratings, movies, train, test, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    out_dir = Path(out_dir)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    counts = ratings["rating"].value_counts().sort_index()
    colors = ["#d1495b" if r < RELEVANT_RATING else "#3b6ea8" for r in counts.index]
    ax.bar(counts.index.astype(str), counts.values, color=colors)
    ax.set_xlabel("Оценка")
    ax.set_ylabel("Число оценок")
    ax.set_title(f"Распределение оценок (синие ≥ {RELEVANT_RATING:g} считаю релевантными)")
    fig.tight_layout()
    fig.savefig(out_dir / "eda_ratings.png", dpi=130)
    plt.close(fig)

    per_user = ratings.groupby("user").size().sort_values(ascending=False).values
    per_item = ratings.groupby("item").size().sort_values(ascending=False).values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    ax1.plot(np.arange(1, len(per_user) + 1), per_user, color="#3b6ea8")
    ax1.set_yscale("log")
    ax1.set_xlabel("Пользователи, по убыванию активности")
    ax1.set_ylabel("Оценок (log)")
    ax1.set_title(f"Активность пользователей (медиана {int(np.median(per_user))})")
    ax2.plot(np.arange(1, len(per_item) + 1), per_item, color="#e08a2c")
    ax2.set_yscale("log")
    ax2.set_xlabel("Фильмы, по убыванию популярности")
    ax2.set_ylabel("Оценок (log)")
    top10 = per_item[: len(per_item) // 10].sum() / per_item.sum()
    ax2.set_title(f"Длинный хвост: 10% фильмов дают {top10:.0%} оценок")
    fig.tight_layout()
    fig.savefig(out_dir / "eda_long_tail.png", dpi=130)
    plt.close(fig)

    users = ratings["user"].unique()
    items = ratings.groupby("item").size().sort_values(ascending=False).index[:600]
    user_index = {u: i for i, u in enumerate(users)}
    item_index = {m: i for i, m in enumerate(items)}
    sub = ratings[ratings["item"].isin(item_index)]
    density = len(ratings) / (len(users) * ratings["item"].nunique())
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.scatter(sub["item"].map(item_index), sub["user"].map(user_index), s=0.15, color="#2f3b45")
    ax.set_xlabel("600 самых популярных фильмов")
    ax.set_ylabel("Пользователи")
    ax.set_title(f"Матрица «пользователь × фильм»: заполнено {density:.1%} всей матрицы "
                 f"(на графике - самая плотная часть)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_dir / "eda_sparsity.png", dpi=130)
    plt.close(fig)

    genres = movies.explode("genre_list").dropna(subset=["genre_list"])
    rated = ratings.merge(genres[["item", "genre_list"]], on="item")
    share = rated["genre_list"].value_counts(normalize=True).head(18)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(share.index[::-1], share.values[::-1], color="#3a9d8f")
    ax.set_xlabel("Доля оценок")
    ax.set_title("Жанры в оценках")
    fig.tight_layout()
    fig.savefig(out_dir / "eda_genres.png", dpi=130)
    plt.close(fig)

    train_n = train.groupby("user").size()
    test_n = test.groupby("user").size()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(train_n, bins=np.logspace(np.log10(train_n.min()), np.log10(train_n.max()), 40),
            color="#3b6ea8", alpha=0.8, label="train")
    ax.hist(test_n, bins=np.logspace(np.log10(max(test_n.min(), 1)), np.log10(test_n.max()), 40),
            color="#e08a2c", alpha=0.8, label="test")
    ax.set_xscale("log")
    ax.set_xlabel("Оценок у пользователя (log)")
    ax.set_ylabel("Пользователей")
    ax.set_title("Временной сплит: последние 20% оценок каждого пользователя - в тест")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "eda_split.png", dpi=130)
    plt.close(fig)


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
