"""
Тематическое моделирование веб-данных (LDA и NMF).

Примеры запуска (из задания):
    python topic_modeling.py --dataset news --method lda --topics auto --output news_topics
    python topic_modeling.py --dataset products --method nmf --topics 15 --output product_categories
    python topic_modeling.py --dataset mixed --method both --topics 10 --compare
"""

import argparse
import csv
import json
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from text_preprocessing import Normalizer, preprocess

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
LR4_OUTPUT = BASE_DIR.parent / "LR4_Scrapy_TwoSpiders_Danilov" / "output"

RANDOM_STATE = 42


# ------------------------------------------------------------------ загрузка данных

def first_existing(*paths):
    for path in paths:
        if not path or not Path(path).exists():
            continue
        if Path(path).suffix == ".json" and not json.loads(Path(path).read_text(encoding="utf-8")):
            continue
        return Path(path)
    return None


def load_news(data_dir):
    path = first_existing(data_dir / "news.json", LR4_OUTPUT / "news.json", DATA_DIR / "lr4_news.json")
    if path is None:
        raise FileNotFoundError("Не найден news.json (запустите collect_data.py или ЛР4)")

    items = json.loads(path.read_text(encoding="utf-8"))
    docs = []
    for item in items:
        parts = [item.get("title", ""), item.get("summary", ""), item.get("text", "")]
        hubs = [h for h in item.get("hubs", []) if h != "*" and not h.lower().startswith("блог компании")]
        parts += hubs + item.get("tags", [])
        docs.append({
            "id": item.get("url", ""),
            "title": item.get("title", ""),
            "text": " ".join(p for p in parts if p),
            "source": item.get("source", "news"),
        })
    return docs, path


def load_products(data_dir):
    path = first_existing(data_dir / "products.json", LR4_OUTPUT / "marketplaces.csv", DATA_DIR / "lr4_marketplaces.csv")
    if path is None:
        raise FileNotFoundError("Не найдены данные товаров (products.json / marketplaces.csv)")

    docs = []
    if path.suffix == ".json":
        for item in json.loads(path.read_text(encoding="utf-8")):
            reviews = " ".join(r["text"] if isinstance(r, dict) else str(r) for r in item.get("reviews", []))
            docs.append({
                "id": item.get("url", ""),
                "title": item.get("title", ""),
                "text": " ".join([item.get("title", ""), item.get("brand", ""), item.get("description", ""), reviews]),
                "source": item.get("source", "products"),
                "n_reviews": len(item.get("reviews", [])),
            })
    else:
        with open(path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f, delimiter=";"):
                docs.append({
                    "id": row.get("url", ""),
                    "title": row.get("title", ""),
                    "text": " ".join([row.get("title", ""), row.get("brand", ""), row.get("description", "")]),
                    "source": row.get("source", "products"),
                    "n_reviews": 0,
                })
    return docs, path


def load_dataset(name, data_dir):
    if name == "news":
        docs, path = load_news(data_dir)
        return docs, [str(path)]
    if name == "products":
        docs, path = load_products(data_dir)
        return docs, [str(path)]
    news, p1 = load_news(data_dir)
    products, p2 = load_products(data_dir)
    for d in news:
        d["dataset"] = "news"
    for d in products:
        d["dataset"] = "products"
    return news + products, [str(p1), str(p2)]


# ------------------------------------------------------------------ coherence c_v

def coherence_cv_own(topics, token_docs, window=110, eps=1e-12):
    """c_v по Röder et al. (2015): булево скользящее окно, NPMI, косвенное
    подтверждение через косинусную меру. Повторяет схему gensim CoherenceModel."""
    vocab = sorted({w for topic in topics for w in topic})
    index = {w: i for i, w in enumerate(vocab)}
    n = len(vocab)
    occ = np.zeros(n)
    co = np.zeros((n, n))
    total = 0

    for tokens in token_docs:
        if not tokens:
            continue
        positions = [index.get(t, -1) for t in tokens]
        if len(positions) <= window:
            windows = [positions]
        else:
            windows = (positions[i:i + window] for i in range(len(positions) - window + 1))
        for win in windows:
            present = sorted({p for p in win if p >= 0})
            total += 1
            if not present:
                continue
            occ[present] += 1
            idx = np.array(present)
            co[np.ix_(idx, idx)] += 1

    if total == 0:
        return 0.0, [0.0] * len(topics)

    p_w = occ / total
    p_ww = co / total
    scores = []
    for topic in topics:
        ids = [index[w] for w in topic]
        k = len(ids)
        npmi = np.zeros((k, k))
        for a in range(k):
            for b in range(k):
                pa, pb, pab = p_w[ids[a]], p_w[ids[b]], p_ww[ids[a], ids[b]]
                pmi = np.log((pab + eps) / (pa * pb + eps))
                npmi[a, b] = pmi / (-np.log(pab + eps))
        topic_vec = npmi.sum(axis=0)
        sims = []
        for a in range(k):
            denom = np.linalg.norm(npmi[a]) * np.linalg.norm(topic_vec)
            sims.append(float(npmi[a] @ topic_vec / denom) if denom > 0 else 0.0)
        scores.append(float(np.mean(sims)))
    return float(np.mean(scores)), scores


def coherence_cv(topics, token_docs, backend="auto"):
    if backend in ("auto", "gensim"):
        try:
            from gensim.corpora import Dictionary
            from gensim.models.coherencemodel import CoherenceModel

            dictionary = Dictionary(token_docs)
            cm = CoherenceModel(topics=topics, texts=token_docs, dictionary=dictionary, coherence="c_v", processes=1)
            per_topic = [float(x) for x in cm.get_coherence_per_topic()]
            return float(np.mean(per_topic)), per_topic, "gensim"
        except ImportError:
            if backend == "gensim":
                raise
    mean, per_topic = coherence_cv_own(topics, token_docs)
    return mean, per_topic, "own"


# ------------------------------------------------------------------ модели

class TopicPipeline:
    def __init__(self, token_docs, n_features, min_df, max_df):
        self.token_docs = token_docs
        self.joined = [" ".join(t) for t in token_docs]
        common = dict(max_features=n_features, min_df=min_df, max_df=max_df,
                      token_pattern=r"(?u)\b[\w\-]{2,}\b")
        self.count_vec = CountVectorizer(**common)
        self.tfidf_vec = TfidfVectorizer(**common)
        self.X_count = self.count_vec.fit_transform(self.joined)
        self.X_tfidf = self.tfidf_vec.fit_transform(self.joined)
        self.terms_count = self.count_vec.get_feature_names_out()
        self.terms_tfidf = self.tfidf_vec.get_feature_names_out()

    def fit(self, method, k):
        start = time.perf_counter()
        if method == "lda":
            model = LatentDirichletAllocation(n_components=k, learning_method="batch", max_iter=50,
                                              random_state=RANDOM_STATE)
            doc_topic = model.fit_transform(self.X_count)
            terms = self.terms_count
        else:
            model = NMF(n_components=k, init="nndsvda", max_iter=1000, random_state=RANDOM_STATE)
            doc_topic = model.fit_transform(self.X_tfidf)
            row_sum = doc_topic.sum(axis=1, keepdims=True)
            doc_topic = np.divide(doc_topic, row_sum, out=np.zeros_like(doc_topic), where=row_sum > 0)
            terms = self.terms_tfidf
        return model, doc_topic, terms, time.perf_counter() - start


def top_words(model, terms, n_top):
    result = []
    for topic in model.components_:
        idx = topic.argsort()[: -n_top - 1: -1]
        weights = topic[idx] / topic.sum()
        result.append([(terms[i], float(w)) for i, w in zip(idx, weights)])
    return result


def select_k(pipeline, method, k_min, k_max, n_top, backend):
    history = []
    for k in range(k_min, k_max + 1):
        model, _, terms, _ = pipeline.fit(method, k)
        topics = [[w for w, _ in t] for t in top_words(model, terms, n_top)]
        score, _, _ = coherence_cv(topics, pipeline.token_docs, backend)
        history.append((k, score))
        print(f"  {method.upper()} K={k:2d}  c_v={score:.4f}")
    best_k = max(history, key=lambda x: x[1])[0]
    return best_k, history


# ------------------------------------------------------------------ визуализация

def plot_top_words(model, feature_names, n_top_words, title, path, display):
    n = model.components_.shape[0]
    cols = 5 if n > 4 else n
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows), sharex=False)
    axes = np.atleast_1d(axes).flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1: -1]
        top_features = [display(feature_names[i]) for i in top_features_ind]
        weights = topic[top_features_ind]
        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7, color="#3b6ea8")
        ax.set_title(f"Тема {topic_idx + 1}", fontsize=13)
        ax.invert_yaxis()
        ax.tick_params(axis="both", labelsize=10)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_coherence(histories, title, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"lda": "#c0504d", "nmf": "#3b6ea8"}
    for method, history in histories.items():
        ks, scores = zip(*history)
        best = int(np.argmax(scores))
        ax.plot(ks, scores, marker="o", color=colors[method], label=method.upper())
        ax.scatter([ks[best]], [scores[best]], s=160, facecolors="none", edgecolors=colors[method], linewidths=2)
        ax.annotate(f"K={ks[best]}\n{scores[best]:.3f}", (ks[best], scores[best]),
                    textcoords="offset points", xytext=(10, -28), color=colors[method])
        ax.set_xticks(list(ks))
    ax.set_xlabel("Число тем K")
    ax.set_ylabel("Coherence c_v")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_doc_topic(doc_topic, title, path, max_docs=60):
    order = np.lexsort((-doc_topic.max(axis=1), doc_topic.argmax(axis=1)))
    step = max(1, len(order) // max_docs)
    sample = doc_topic[order[::step]]
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(sample, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(doc_topic.shape[1]))
    ax.set_xticklabels([f"T{i + 1}" for i in range(doc_topic.shape[1])])
    ax.set_ylabel(f"Документы (каждый {step}-й, отсортированы по главной теме)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Доля темы в документе")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_topic_share(doc_topic, labels, title, path):
    share = doc_topic.mean(axis=0)
    dominant = np.bincount(doc_topic.argmax(axis=1), minlength=doc_topic.shape[1])
    x = np.arange(len(share))
    fig, ax1 = plt.subplots(figsize=(max(9, 0.8 * len(share) + 3), 6))
    ax1.bar(x, share, color="#3b6ea8", label="Средняя доля темы")
    ax1.set_ylabel("Средняя доля в корпусе")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"T{i + 1}: {labels[i]}" for i in x], fontsize=9, rotation=40, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, dominant, color="#c0504d", marker="o", label="Документов с главной темой")
    ax2.set_ylabel("Документов, где тема главная")
    ax1.set_title(title)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_topic_similarity(model, title, path):
    comps = model.components_ / np.linalg.norm(model.components_, axis=1, keepdims=True)
    sim = comps @ comps.T
    k = sim.shape[0]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sim, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels([f"T{i + 1}" for i in range(k)])
    ax.set_yticklabels([f"T{i + 1}" for i in range(k)])
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Косинусная близость")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_wordclouds(topics, title, path, display):
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("  wordcloud не установлен - облака слов пропущены")
        return False
    k = len(topics)
    cols = 5 if k > 4 else k
    rows = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows))
    axes = np.atleast_1d(axes).flatten()
    for i, topic in enumerate(topics):
        freqs = {display(w): w_ for w, w_ in topic}
        wc = WordCloud(width=500, height=380, background_color="white", colormap="viridis",
                       random_state=RANDOM_STATE).generate_from_frequencies(freqs)
        axes[i].imshow(wc, interpolation="bilinear")
        axes[i].set_title(f"Тема {i + 1}")
        axes[i].axis("off")
    for ax in axes[k:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def plot_compare(results, title, path):
    methods = list(results)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    colors = ["#c0504d", "#3b6ea8"]
    metrics = [
        ("coherence", "Coherence c_v"),
        ("overlap", "Пересечение топ-слов тем (Жаккар)"),
        ("fit_seconds", "Время обучения, с"),
    ]
    for ax, (key, label) in zip(axes, metrics):
        values = [results[m][key] for m in methods]
        bars = ax.bar([m.upper() for m in methods], values, color=colors[: len(methods)])
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.3f}", ha="center", va="bottom")
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ основной сценарий

def mean_jaccard_overlap(topics):
    sets = [set(w for w, _ in t) for t in topics]
    values = [len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1:]]
    return float(np.mean(values)) if values else 0.0


def run_method(method, k, pipeline, docs, args, out_prefix, display):
    model, doc_topic, terms, fit_seconds = pipeline.fit(method, k)
    topics = top_words(model, terms, args.n_top_words)
    topic_terms = [[w for w, _ in t] for t in topics]
    mean_cv, per_topic_cv, backend = coherence_cv(topic_terms, pipeline.token_docs, args.coherence)
    labels = [display(t[0][0]) for t in topics]
    name = method.upper()

    plot_top_words(model, terms, args.n_top_words,
                   f"{name}: топ-{args.n_top_words} слов тем ({args.dataset}, K={k})",
                   f"{out_prefix}_{method}_top_words.png", display)
    plot_doc_topic(doc_topic, f"{name}: распределение тем по документам",
                   f"{out_prefix}_{method}_doc_topic.png")
    plot_topic_share(doc_topic, labels, f"{name}: доли тем в корпусе",
                     f"{out_prefix}_{method}_topic_share.png")
    plot_topic_similarity(model, f"{name}: близость тем", f"{out_prefix}_{method}_topic_similarity.png")
    plot_wordclouds(topics, f"{name}: облака слов", f"{out_prefix}_{method}_wordclouds.png", display)

    with open(f"{out_prefix}_{method}_doc_topics.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["doc_id", "title", "dominant_topic", "dominant_share"] + [f"T{i + 1}" for i in range(k)])
        for doc, row in zip(docs, doc_topic):
            writer.writerow([doc["id"], doc["title"], int(row.argmax()) + 1, round(float(row.max()), 4)]
                            + [round(float(v), 4) for v in row])

    topic_json = []
    for i, topic in enumerate(topics):
        members = np.where(doc_topic.argmax(axis=1) == i)[0]
        examples = sorted(members, key=lambda j: -doc_topic[j, i])[:3]
        topic_json.append({
            "topic": i + 1,
            "coherence_cv": round(per_topic_cv[i], 4),
            "share": round(float(doc_topic[:, i].mean()), 4),
            "documents": int(len(members)),
            "keywords": [{"term": display(w), "weight": round(wt, 4)} for w, wt in topic],
            "example_titles": [docs[j]["title"] for j in examples],
        })

    summary = {
        "method": method,
        "k": k,
        "coherence": round(mean_cv, 4),
        "coherence_backend": backend,
        "overlap": round(mean_jaccard_overlap(topics), 4),
        "fit_seconds": round(fit_seconds, 3),
        "topics": topic_json,
    }
    Path(f"{out_prefix}_{method}_topics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{name}, K={k}: coherence c_v = {mean_cv:.4f} ({backend}), пересечение тем = {summary['overlap']:.3f}")
    for t in topic_json:
        words = ", ".join(f"{kw['term']}({kw['weight']:.3f})" for kw in t["keywords"][:8])
        print(f"  T{t['topic']:<2} c_v={t['coherence_cv']:.3f} docs={t['documents']:<3} {words}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Тематическое моделирование веб-данных: LDA и NMF")
    parser.add_argument("--dataset", choices=["news", "products", "mixed"], required=True)
    parser.add_argument("--method", choices=["lda", "nmf", "both"], default="both")
    parser.add_argument("--topics", default="auto", help="число тем или auto (подбор по coherence)")
    parser.add_argument("--output", default=None, help="префикс файлов результата")
    parser.add_argument("--compare", action="store_true", help="сравнить LDA и NMF на одном датасете")
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--k-max", type=int, default=20)
    parser.add_argument("--n-features", type=int, default=1000)
    parser.add_argument("--n-top-words", type=int, default=10)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-df", type=float, default=0.6)
    parser.add_argument("--normalizer", choices=["auto", "lemma", "stem"], default="auto")
    parser.add_argument("--coherence", choices=["auto", "gensim", "own"], default="auto")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    return parser.parse_args()


def main():
    args = parse_args()
    methods = ["lda", "nmf"] if (args.method == "both" or args.compare) else [args.method]
    output = args.output or f"{args.dataset}_topics"
    RESULTS_DIR.mkdir(exist_ok=True)
    out_prefix = str(RESULTS_DIR / output)

    docs, sources = load_dataset(args.dataset, Path(args.data_dir))
    sources = [os.path.relpath(s, BASE_DIR) for s in sources]
    normalizer = Normalizer(args.normalizer)
    token_docs = preprocess([d["text"] for d in docs], normalizer)
    keep = [i for i, t in enumerate(token_docs) if len(t) >= 2]
    docs = [docs[i] for i in keep]
    token_docs = [token_docs[i] for i in keep]

    print(f"Датасет: {args.dataset}, документов: {len(docs)} (источник: {', '.join(sources)})")
    print(f"Нормализация: {normalizer.mode}, средняя длина документа: {np.mean([len(t) for t in token_docs]):.1f} токенов")

    pipeline = TopicPipeline(token_docs, args.n_features, args.min_df, args.max_df)
    print(f"Словарь: {len(pipeline.terms_count)} терминов")

    chosen_k, histories = {}, {}
    if args.topics == "auto":
        k_max = min(args.k_max, len(docs) - 1)
        for method in methods:
            print(f"\nПодбор K для {method.upper()} по coherence c_v ({args.k_min}..{k_max}):")
            chosen_k[method], histories[method] = select_k(pipeline, method, args.k_min, k_max,
                                                           args.n_top_words, args.coherence)
        plot_coherence(histories, f"Подбор числа тем по coherence ({args.dataset})", f"{out_prefix}_coherence.png")
    else:
        for method in methods:
            chosen_k[method] = int(args.topics)

    results = {}
    for method in methods:
        results[method] = run_method(method, chosen_k[method], pipeline, docs, args, out_prefix, normalizer.display)

    if len(methods) == 2:
        plot_compare(results, f"Сравнение LDA и NMF ({args.dataset})", f"{out_prefix}_compare.png")
        print("\nСравнение методов:")
        print(f"  {'метод':<6}{'K':>4}{'c_v':>9}{'пересеч.':>10}{'время, с':>10}")
        for m, r in results.items():
            print(f"  {m.upper():<6}{r['k']:>4}{r['coherence']:>9.4f}{r['overlap']:>10.3f}{r['fit_seconds']:>10.3f}")

    report = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "sources": sources,
        "documents": len(docs),
        "vocabulary": int(len(pipeline.terms_count)),
        "normalizer": normalizer.mode,
        "coherence_history": {m: [[k, round(s, 4)] for k, s in h] for m, h in histories.items()},
        "results": {m: {k: v for k, v in r.items() if k != "topics"} for m, r in results.items()},
    }
    Path(f"{out_prefix}_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nРезультаты сохранены в results/ с префиксом '{output}'")


if __name__ == "__main__":
    main()
