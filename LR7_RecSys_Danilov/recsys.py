# CLI рекомендательной системы: обучаю модели, считаю метрики и сохраняю рекомендации

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from data import RELEVANT_RATING, describe, load, plot_eda, save_json, temporal_split
from evaluation import MODEL_NAMES, Evaluator, plot_results, segment_of
from models import MODELS


def parse_args():
    parser = argparse.ArgumentParser(description="Рекомендательная система на MovieLens (ЛР7)")
    parser.add_argument("--dataset", default="movielens", choices=["movielens"])
    parser.add_argument("--model", nargs="+", default=["all"], choices=list(MODELS) + ["all"])
    parser.add_argument("--k", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--output", default="results")
    parser.add_argument("--evaluate", action="store_true", help="посчитать метрики на тесте")
    parser.add_argument("--eda", action="store_true", help="сохранить графики разведочного анализа")
    parser.add_argument("--sample-users", type=int, default=20, help="сколько пользователей сохранить в recs.json")
    parser.add_argument("--factors", type=int, default=None, help="число латентных факторов для SVD/ALS/BPR")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_model(name, args):
    cls = MODELS[name]
    if args.factors and name in ("svd", "als", "bpr"):
        return cls(factors=args.factors, seed=args.seed)
    if name in ("svd", "als", "bpr"):
        return cls(seed=args.seed)
    return cls()


def pick_sample_users(evaluator, train, n, seed):
    """Беру двух пользователей для ручного разбора (новичка и активного) и ещё случайных."""
    rng = random.Random(seed)
    users = evaluator.users
    light = [u for u in users if evaluator.train_size.get(u, 0) <= 30 and len(evaluator.relevant[u]) >= 3]
    heavy = [u for u in users if evaluator.train_size.get(u, 0) > 150 and len(evaluator.relevant[u]) >= 5]
    chosen = []
    for pool in (light, heavy):
        if pool:
            chosen.append(rng.choice(pool))
    rest = [u for u in users if u not in chosen]
    chosen += rng.sample(rest, min(max(n - len(chosen), 0), len(rest)))
    return chosen


def describe_items(items, movies, relevant=None, model=None, user=None):
    out = []
    for item in items:
        row = movies.loc[item]
        entry = {"item": int(item), "title": row["title"], "genres": row["genres"]}
        if relevant is not None:
            entry["hit"] = item in relevant
        if model is not None:
            reason = model.explain(user, item)
            if isinstance(reason, (int, np.integer)) and reason in movies.index:
                reason = f"похож на «{movies.loc[reason, 'title']}»"
            if reason:
                entry["reason"] = reason
        out.append(entry)
    return out


def main():
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    np.random.seed(args.seed)
    random.seed(args.seed)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    ratings, movies, tags = load(args.dataset)
    train, test = temporal_split(ratings)
    train_ts = set(zip(train["user"], train["item"]))
    if any(pair in train_ts for pair in zip(test["user"], test["item"])):
        raise SystemExit("В train и test попала одна и та же пара пользователь-фильм")
    leak = (train.groupby("user")["timestamp"].max() > test.groupby("user")["timestamp"].min()).sum()
    tag_cut = test.groupby("user")["timestamp"].min()
    tags = tags[tags["timestamp"] < tags["user"].map(tag_cut).fillna(np.inf)]

    stats = describe(ratings, movies, tags, train, test)
    stats["users_with_time_overlap"] = int(leak)
    save_json(out_dir / "eda.json", stats)
    print(f"Данные: {stats['ratings']} оценок, {stats['users']} пользователей, {stats['items_in_catalog']} фильмов, "
          f"разреженность {stats['sparsity']:.2%}")
    print(f"Сплит: train {stats['train_ratings']}, test {stats['test_ratings']} "
          f"(релевантных {stats['test_relevant']}, из них на фильмы без оценок в train {stats['test_relevant_cold_items']})")
    if args.eda:
        plot_eda(ratings, movies, train, test, out_dir)

    names = list(MODELS) if "all" in args.model else list(dict.fromkeys(args.model))
    if "popular" not in names:
        print("Добавляю бейзлайн popular: без него сравнение не засчитывается")
        names.insert(0, "popular")

    evaluator = Evaluator(train, test, movies, args.k)
    sample = pick_sample_users(evaluator, train, args.sample_users, args.seed)
    movies_idx = movies.set_index("item")

    metric_rows, beyond_rows, timing, extra = [], [], [], {}
    recs_json = {"split": "последние 20% оценок каждого пользователя по времени",
                 "relevant": f"оценка в тесте ≥ {RELEVANT_RATING:g}", "users": {}}
    for user in sample:
        liked = train[(train["user"] == user)].sort_values(["rating", "timestamp"], ascending=False).head(8)
        recs_json["users"][str(user)] = {
            "train_ratings": evaluator.train_size.get(user, 0),
            "segment": segment_of(evaluator.train_size.get(user, 0)),
            "liked_in_train": [{"title": movies_idx.loc[i, "title"], "genres": movies_idx.loc[i, "genres"],
                                "rating": float(r)} for i, r in zip(liked["item"], liked["rating"])],
            "relevant_in_test": [movies_idx.loc[i, "title"] for i in sorted(evaluator.relevant.get(user, []))],
            "recommendations": {},
        }

    for name in names:
        started = time.time()
        try:
            model = build_model(name, args).fit(train, movies, tags)
        except ImportError as error:
            print(f"[{name}] пропущена: не установлена библиотека ({error})")
            continue
        fit_time = time.time() - started
        started = time.time()
        users = evaluator.users if args.evaluate else sample
        recs = {u: model.recommend(u, evaluator.k_max) for u in users}
        rec_time = time.time() - started
        timing.append({"model": name, "fit_seconds": round(fit_time, 2), "recommend_seconds": round(rec_time, 2)})
        print(f"[{MODEL_NAMES[name]}] обучение {fit_time:.1f} с, рекомендации {rec_time:.1f} с")

        if hasattr(model, "rating_metrics"):
            extra[name] = model.rating_metrics(test)
            print(f"   RMSE на тесте {extra[name]['rmse']:.3f}, MAE {extra[name]['mae']:.3f}")

        for user in sample:
            items = recs.get(user) or model.recommend(user, evaluator.k_max)
            recs_json["users"][str(user)]["recommendations"][name] = describe_items(
                items[:10], movies_idx, evaluator.relevant.get(user, set()), model, user)

        if args.evaluate:
            rows = evaluator.evaluate(name, recs)
            metric_rows += rows
            for k in args.k:
                beyond_rows.append(evaluator.beyond_accuracy(name, recs, k))
            main = {r["metric"]: r["value"] for r in rows if r["k"] == 10 and r["segment"] == "все"} or \
                   {r["metric"]: r["value"] for r in rows if r["k"] == evaluator.ks[0] and r["segment"] == "все"}
            print("   " + "  ".join(f"{m}={v:.4f}" for m, v in main.items()))

    save_json(out_dir / "recs.json", recs_json)
    pd.DataFrame(timing).to_csv(out_dir / "timing.csv", index=False)
    if extra:
        save_json(out_dir / "rating_metrics.json", extra)

    if args.evaluate and metric_rows:
        metrics = pd.DataFrame(metric_rows)
        metrics.to_csv(out_dir / "metrics.csv", index=False)
        beyond = pd.DataFrame(beyond_rows)
        beyond.to_csv(out_dir / "beyond_accuracy.csv", index=False)
        k_main = 10 if 10 in args.k else evaluator.ks[0]
        plot_results(metrics, beyond, out_dir, k_main)

        table = metrics[(metrics["segment"] == "все") & (metrics["k"] == k_main)].pivot(
            index="model", columns="metric", values="value").reindex(names).dropna(how="all")
        b = beyond[beyond["k"] == k_main].set_index("model")
        table = table.join(b[["coverage_catalog", "diversity_ild", "novelty_bits", "bubble_share"]])
        print(f"\nИтог при K={k_main} ({len(evaluator.users)} пользователей с релевантными фильмами в тесте):")
        print(table[["precision", "recall", "map", "ndcg", "hit_rate", "coverage_catalog", "diversity_ild",
                     "novelty_bits", "bubble_share"]].round(4).to_string())
    print(f"\nРезультаты сохранены в {out_dir}/")


if __name__ == "__main__":
    main()
