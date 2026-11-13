# метрики ранжирования, покрытие, разнообразие, пузырь фильтров и графики

from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from data import RELEVANT_RATING

SEGMENTS = [("новые (≤30 оценок)", 0, 30), ("средние (31-150)", 31, 150), ("активные (>150)", 151, 10 ** 9)]
MODEL_COLORS = {"popular": "#9aa5ae", "content": "#3a9d8f", "svd": "#e08a2c", "als": "#3b6ea8", "bpr": "#8a6bbf"}
MODEL_NAMES = {"popular": "Популярное", "content": "Контентная TF-IDF", "svd": "SVD (surprise)",
               "als": "ALS (implicit)", "bpr": "BPR (implicit)"}


def ranking_metrics(recs, relevant, k):
    top = recs[:k]
    hits = [1 if item in relevant else 0 for item in top]
    n_hits = sum(hits)
    precision = n_hits / k
    recall = n_hits / len(relevant)
    ap, found = 0.0, 0
    for i, h in enumerate(hits, start=1):
        if h:
            found += 1
            ap += found / i
    ap /= min(len(relevant), k)
    dcg = sum(h / np.log2(i + 1) for i, h in enumerate(hits, start=1))
    idcg = sum(1 / np.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return {"precision": precision, "recall": recall, "map": ap, "ndcg": dcg / idcg, "hit_rate": float(n_hits > 0)}


def segment_of(n_train):
    for name, low, high in SEGMENTS:
        if low <= n_train <= high:
            return name
    return SEGMENTS[-1][0]


class Evaluator:
    def __init__(self, train, test, movies, ks):
        self.ks = sorted(ks)
        self.k_max = max(self.ks)
        self.movies = movies.set_index("item")
        relevant = test[test["rating"] >= RELEVANT_RATING]
        self.relevant = relevant.groupby("user")["item"].apply(set).to_dict()
        self.users = sorted(self.relevant)
        self.train_size = train.groupby("user").size().to_dict()
        self.catalog = len(movies)
        self.train_items = set(train["item"])

        users_per_item = train.groupby("item")["user"].nunique()
        n_users = train["user"].nunique()
        self.self_info = {item: -np.log2((n + 1) / (n_users + 1)) for item, n in users_per_item.items()}
        self.default_info = -np.log2(1 / (n_users + 1))
        head = users_per_item.sort_values(ascending=False)
        self.head_items = set(head.index[: max(1, len(head) // 10)])

        genres = sorted({g for gl in movies["genre_list"] for g in gl})
        self.genre_index = {g: i for i, g in enumerate(genres)}
        self.genre_vec = {}
        for row in movies.itertuples():
            v = np.zeros(len(genres))
            for g in row.genre_list:
                v[self.genre_index[g]] = 1
            n = np.linalg.norm(v)
            self.genre_vec[row.item] = v / n if n else v

        liked = train[train["rating"] >= RELEVANT_RATING].merge(movies[["item", "genre_list"]], on="item")
        self.user_top_genres = {}
        for user, group in liked.groupby("user"):
            counts = Counter(g for gl in group["genre_list"] for g in gl)
            self.user_top_genres[user] = {g for g, _ in counts.most_common(3)}
        test_genres = relevant.merge(movies[["item", "genre_list"]], on="item")
        self.actual_bubble = self.bubble_share(
            {u: list(g["item"]) for u, g in test_genres.groupby("user")})

    def genres_of(self, item):
        return self.movies.at[item, "genre_list"] if item in self.movies.index else []

    def intra_list_diversity(self, items):
        if len(items) < 2:
            return 0.0
        dists = [1 - float(self.genre_vec[a] @ self.genre_vec[b]) for a, b in combinations(items, 2)]
        return float(np.mean(dists))

    def bubble_share(self, recs_by_user):
        shares = []
        for user, items in recs_by_user.items():
            top = self.user_top_genres.get(user)
            if not top or not items:
                continue
            inside = [bool(set(self.genres_of(i)) & top) for i in items]
            shares.append(np.mean(inside))
        return float(np.mean(shares)) if shares else 0.0

    def evaluate(self, name, recs_by_user):
        rows, per_user = [], []
        for user in self.users:
            recs = recs_by_user.get(user, [])
            for k in self.ks:
                m = ranking_metrics(recs, self.relevant[user], k)
                per_user.append({"model": name, "user": user, "k": k,
                                 "segment": segment_of(self.train_size.get(user, 0)), **m})
        per_user = pd.DataFrame(per_user)
        for (k, segment), group in [((k, "все"), g) for k, g in per_user.groupby("k")] + \
                [((k, s), g) for (k, s), g in per_user.groupby(["k", "segment"])]:
            for metric in ("precision", "recall", "map", "ndcg", "hit_rate"):
                values = group[metric].to_numpy()
                rows.append({"model": name, "k": k, "segment": segment, "metric": metric,
                             "value": round(float(values.mean()), 5),
                             "ci95": round(float(1.96 * values.std(ddof=1) / np.sqrt(len(values))), 5)
                             if len(values) > 1 else 0.0,
                             "users": len(values)})
        return rows

    def beyond_accuracy(self, name, recs_by_user, k):
        lists = {u: recs_by_user.get(u, [])[:k] for u in self.users}
        recommended = Counter(i for items in lists.values() for i in items)
        all_items = [i for items in lists.values() for i in items]
        return {
            "model": name, "k": k,
            "coverage_catalog": round(len(recommended) / self.catalog, 4),
            "unique_items": len(recommended),
            "diversity_ild": round(float(np.mean([self.intra_list_diversity(x) for x in lists.values()])), 4),
            "novelty_bits": round(float(np.mean([self.self_info.get(i, self.default_info) for i in all_items])), 3),
            "long_tail_share": round(float(np.mean([i not in self.head_items for i in all_items])), 4),
            "cold_items_share": round(float(np.mean([i not in self.train_items for i in all_items])), 4),
            "top1_item_share": round(recommended.most_common(1)[0][1] / max(len(lists), 1), 4) if recommended else 0,
            "bubble_share": round(self.bubble_share(lists), 4),
            "bubble_share_actual": round(self.actual_bubble, 4),
        }


def plot_results(metrics, beyond, out_dir, k_main=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    out_dir = Path(out_dir)
    models = list(dict.fromkeys(metrics["model"]))
    names = [MODEL_NAMES.get(m, m) for m in models]
    colors = [MODEL_COLORS.get(m, "#777") for m in models]
    allm = metrics[(metrics["segment"] == "все")]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    for ax, metric, title in zip(axes, ["precision", "recall", "map", "ndcg"],
                                 [f"Precision@{k_main}", f"Recall@{k_main}", f"MAP@{k_main}", f"NDCG@{k_main}"]):
        sub = allm[(allm["k"] == k_main) & (allm["metric"] == metric)].set_index("model").reindex(models)
        bars = ax.bar(range(len(models)), sub["value"], yerr=sub["ci95"], color=colors, capsize=3)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.set_title(title)
        for b, v, ci in zip(bars, sub["value"], sub["ci95"]):
            ax.text(b.get_x() + b.get_width() / 2 + 0.22, v + ci, f"{v:.3f}", ha="left", va="bottom", fontsize=8)
    fig.suptitle("Качество ранжирования (все пользователи, усы - 95% доверительный интервал)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_main.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, metric in zip(axes, ["ndcg", "recall"]):
        for m, n, c in zip(models, names, colors):
            sub = allm[(allm["model"] == m) & (allm["metric"] == metric)].sort_values("k")
            ax.plot(sub["k"], sub["value"], marker="o", color=c, label=n)
        ax.set_xlabel("K")
        ax.set_title(f"{metric.upper() if metric == 'ndcg' else 'Recall'}@K")
        ax.set_xticks(sorted(allm["k"].unique()))
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_by_k.png", dpi=130)
    plt.close(fig)

    seg = metrics[(metrics["k"] == k_main) & (metrics["metric"] == "ndcg") & (metrics["segment"] != "все")]
    segments = [s for s, _, _ in SEGMENTS if s in set(seg["segment"])]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    width = 0.8 / len(models)
    for j, (m, n, c) in enumerate(zip(models, names, colors)):
        vals = [seg[(seg["model"] == m) & (seg["segment"] == s)]["value"].mean() for s in segments]
        ax.bar(np.arange(len(segments)) + j * width - 0.4 + width / 2, vals, width, color=c, label=n)
    users = [int(seg[seg["segment"] == s]["users"].iloc[0]) for s in segments]
    ax.set_xticks(range(len(segments)))
    ax.set_xticklabels([f"{s}\n{u} польз." for s, u in zip(segments, users)])
    ax.set_ylabel(f"NDCG@{k_main}")
    ax.set_title("Качество по сегментам пользователей (по числу оценок в train)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_segments.png", dpi=130)
    plt.close(fig)

    b = beyond[beyond["k"] == k_main].set_index("model").reindex(models)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    panels = [("coverage_catalog", "Покрытие каталога", "{:.1%}"), ("diversity_ild", "Разнообразие списка (ILD)", "{:.2f}"),
              ("novelty_bits", "Новизна, бит", "{:.1f}"), ("long_tail_share", "Доля длинного хвоста", "{:.0%}")]
    for ax, (col, title, fmt) in zip(axes, panels):
        bars = ax.bar(range(len(models)), b[col], color=colors)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.set_title(title)
        for bar, v in zip(bars, b[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"Покрытие и разнообразие выдачи (top-{k_main})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "beyond_accuracy.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.4))
    bars = ax.bar(range(len(models)), b["bubble_share"], color=colors)
    actual = float(b["bubble_share_actual"].iloc[0])
    ax.axhline(actual, color="#d1495b", ls="--", label=f"что пользователи реально оценили в тесте: {actual:.0%}")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Доля рекомендаций")
    ax.set_title("Пузырь фильтров: доля рекомендаций в трёх любимых жанрах пользователя")
    for bar, v in zip(bars, b["bubble_share"]):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.0%}", ha="center", va="bottom", fontsize=9)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "filter_bubble.png", dpi=130)
    plt.close(fig)
