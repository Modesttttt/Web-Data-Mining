# группирую ролики по темам через KMeans и NMF

import argparse
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from common import (
    OBJECT_NAMES_RU, RESULTS_DIR, SCENE_MIN_SCORE, YOLO_CONF, ensure_cwd, load_index, write_csv, write_json,
)
from text_preprocessing import Normalizer, preprocess

RANDOM_STATE = 42

SPOKEN_STOPWORDS = """
вот ну короче типа вообще кстати конечно например именно наверное вроде собственно значит
давать давай давайте ребята друзья привет всем всех видео ролик ролика канал канале подписка
подписаться лайк комментарий комментарии сегодня сейчас потом тоже также просто очень
какой-то что-то кто-то где-то как-то почему-то вещь штука момент раз случай плюс минус
говорить сказать смотреть посмотреть показать хотеть мочь знать думать понимать видеть
делать сделать быть стать пойти идти взять дать получить работать
линукс linux система системы дистрибутив дистрибутивы
""".split()

PLOT_PALETTE = ["#3b6ea8", "#e08a2c", "#5bb04a", "#d1495b", "#8a6bbf", "#3a9d8f"]


def coherence_cv(topics, token_docs, window=110, eps=1e-12):
    """Считаю coherence c_v по Röder et al. (2015), так же, как делал в ЛР5."""
    vocab = sorted({w for topic in topics for w in topic})
    index = {w: i for i, w in enumerate(vocab)}
    n = len(vocab)
    occ, co, total = np.zeros(n), np.zeros((n, n)), 0
    for tokens in token_docs:
        positions = [index.get(t, -1) for t in tokens]
        windows = [positions] if len(positions) <= window else (
            positions[i:i + window] for i in range(len(positions) - window + 1))
        for win in windows:
            present = sorted({p for p in win if p >= 0})
            total += 1
            if present:
                occ[present] += 1
                idx = np.array(present)
                co[np.ix_(idx, idx)] += 1
    if total == 0:
        return 0.0, [0.0] * len(topics)
    p_w, p_ww = occ / total, co / total
    scores = []
    for topic in topics:
        ids = [index[w] for w in topic]
        k = len(ids)
        npmi = np.zeros((k, k))
        for a in range(k):
            for b in range(k):
                pab = p_ww[ids[a], ids[b]]
                npmi[a, b] = np.log((pab + eps) / (p_w[ids[a]] * p_w[ids[b]] + eps)) / (-np.log(pab + eps))
        topic_vec = npmi.sum(axis=0)
        sims = [float(npmi[a] @ topic_vec / (np.linalg.norm(npmi[a]) * np.linalg.norm(topic_vec)))
                if np.linalg.norm(npmi[a]) * np.linalg.norm(topic_vec) > 0 else 0.0 for a in range(k)]
        scores.append(float(np.mean(sims)))
    return float(np.mean(scores)), scores


def visual_tokens(video, weight=30):
    """Превращаю сцены и объекты в слова: чем дольше что-то было в кадре, тем больше повторов."""
    tokens = []
    frames = max(len(video["frames"]), 1)
    scenes = Counter(s["scene"] for s in video["scenes"] if float(s["score"]) >= SCENE_MIN_SCORE)
    for scene, n in scenes.items():
        tokens += ["сцена:" + scene.replace(" ", "_")] * int(round(weight * n / frames))
    seen = {}
    for d in video["detections"]:
        if float(d["conf"]) >= YOLO_CONF:
            seen.setdefault(d["class"], set()).add(d["t"])
    for cls, ts in seen.items():
        tokens += ["объект:" + OBJECT_NAMES_RU.get(cls, cls).replace(" ", "_")] * int(round(weight * len(ts) / frames))
    return tokens


def build_documents(index, use_visual=True):
    normalizer = Normalizer()
    texts = []
    for v in index:
        info = v["info"]
        parts = [info.get("title") or "", (info.get("description") or "")[:1500], " ".join(info.get("tags") or []),
                 v["transcript"].get("text", "")]
        texts.append(" ".join(parts))
    token_docs = preprocess(texts, normalizer, extra_stopwords=SPOKEN_STOPWORDS)
    if use_visual:
        token_docs = [tokens + visual_tokens(v) for tokens, v in zip(token_docs, index)]
    return token_docs, normalizer


def cluster_terms(X, labels, terms, n=8):
    result = {}
    for c in sorted(set(labels)):
        centroid = np.asarray(X[labels == c].mean(axis=0)).ravel()
        result[int(c)] = [terms[i] for i in centroid.argsort()[::-1][:n]]
    return result


def main():
    parser = argparse.ArgumentParser(description="Тематическая группировка роликов")
    parser.add_argument("--k-max", type=int, default=5)
    parser.add_argument("--no-visual", action="store_true", help="не добавлять объекты и сцены в документы")
    args = parser.parse_args()

    ensure_cwd()
    index = load_index()
    index = [v for v in index if v["transcript"].get("text")]
    if len(index) < 4:
        raise SystemExit("Нужно хотя бы 4 ролика с транскриптом")
    names = [v["distro"] for v in index]
    token_docs, normalizer = build_documents(index, not args.no_visual)
    display = normalizer.display

    vectorizer = TfidfVectorizer(analyzer=lambda doc: doc, min_df=2, max_df=0.8, sublinear_tf=True)
    try:
        X = vectorizer.fit_transform(token_docs)
    except ValueError:
        print("Слишком мало общих слов при min_df=2, max_df=0.8 — беру все слова")
        vectorizer = TfidfVectorizer(analyzer=lambda doc: doc, min_df=1, max_df=1.0, sublinear_tf=True)
        X = vectorizer.fit_transform(token_docs)
    terms = vectorizer.get_feature_names_out()
    print(f"Роликов: {len(index)}, терминов в словаре: {len(terms)} (нормализация: {normalizer.mode})")

    k_values = list(range(2, min(args.k_max, len(index) - 1) + 1))

    # Сначала KMeans, число кластеров выбираю по silhouette
    km_scores, km_labels = {}, {}
    for k in k_values:
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20).fit_predict(X)
        km_labels[k] = labels
        km_scores[k] = float(silhouette_score(X, labels, metric="cosine")) if len(set(labels)) > 1 else -1
        print(f"  KMeans k={k}: silhouette={km_scores[k]:.3f}")
    best_k = max(km_scores, key=km_scores.get)
    labels = km_labels[best_k]
    terms_by_cluster = cluster_terms(X, labels, terms)

    # Потом NMF, число тем выбираю по coherence
    nmf_scores, nmf_models = {}, {}
    for k in k_values:
        model = NMF(n_components=k, init="nndsvda", random_state=RANDOM_STATE, max_iter=1000)
        W = model.fit_transform(X)
        top = [[terms[i] for i in comp.argsort()[::-1][:10]] for comp in model.components_]
        nmf_scores[k], _ = coherence_cv(top, token_docs)
        nmf_models[k] = (model, W)
        print(f"  NMF k={k}: coherence c_v={nmf_scores[k]:.3f}")
    best_t = max(nmf_scores, key=nmf_scores.get)
    model, W = nmf_models[best_t]
    shares = W / np.where(W.sum(axis=1, keepdims=True) > 0, W.sum(axis=1, keepdims=True), 1)
    topics = []
    for i, comp in enumerate(model.components_):
        idx = comp.argsort()[::-1][:10]
        topics.append({"topic": i + 1, "words": [{"term": display(terms[j]), "weight": round(float(comp[j] / comp.sum()), 4)}
                                                 for j in idx]})

    rows = []
    for i, v in enumerate(index):
        row = {"distro": v["distro"], "id": v["id"], "cluster": int(labels[i]) + 1,
               "main_topic": int(shares[i].argmax()) + 1}
        row.update({f"T{t + 1}": round(float(shares[i, t]), 3) for t in range(best_t)})
        rows.append(row)
    write_csv(RESULTS_DIR / "clusters.csv", rows, list(rows[0]))
    write_json(RESULTS_DIR / "topics.json", {
        "documents": len(index), "vocabulary": len(terms), "normalizer": normalizer.mode,
        "kmeans": {"silhouette": km_scores, "best_k": best_k,
                   "clusters": {str(c + 1): {"videos": [n for n, l in zip(names, labels) if l == c],
                                              "terms": [display(t) for t in terms_by_cluster[c]]}
                                for c in sorted(set(labels))}},
        "nmf": {"coherence": nmf_scores, "best_k": best_t, "topics": topics},
    })

    print(f"\nKMeans: k={best_k} (silhouette {km_scores[best_k]:.3f})")
    for c in sorted(set(labels)):
        print(f"  кластер {c + 1}: {', '.join(n for n, l in zip(names, labels) if l == c)}"
              f"  — {', '.join(display(t) for t in terms_by_cluster[c][:6])}")
    print(f"NMF: {best_t} темы (c_v {nmf_scores[best_t]:.3f})")
    for t in topics:
        print(f"  T{t['topic']}: {', '.join(w['term'] for w in t['words'][:8])}")

    plot_k_selection(km_scores, nmf_scores, best_k, best_t)
    plot_map(X, names, labels)
    plot_topic_words(topics)
    plot_doc_topics(names, shares)


def plot_k_selection(km_scores, nmf_scores, best_k, best_t):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for ax, scores, best, title, ylabel, color in (
        (ax1, km_scores, best_k, "KMeans: качество кластеров", "silhouette (cosine)", "#3b6ea8"),
        (ax2, nmf_scores, best_t, "NMF: согласованность тем", "coherence c_v", "#e08a2c"),
    ):
        ks = list(scores)
        ax.plot(ks, [scores[k] for k in ks], marker="o", color=color)
        ax.scatter([best], [scores[best]], s=150, facecolors="none", edgecolors=color, linewidths=2)
        ax.set_xticks(ks)
        ax.set_xlabel("Число кластеров / тем")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "topics_k_selection.png", dpi=120)
    plt.close(fig)


def plot_map(X, names, labels):
    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X.toarray())
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for c in sorted(set(labels)):
        idx = labels == c
        ax.scatter(coords[idx, 0], coords[idx, 1], s=180, color=PLOT_PALETTE[c % len(PLOT_PALETTE)],
                   edgecolor="black", linewidth=0.6, label=f"Кластер {c + 1}")
    for (x, y), name in zip(coords, names):
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(8, 6), fontsize=10)
    ax.set_title("Ролики в пространстве TF-IDF (PCA, 2 компоненты)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "topics_map.png", dpi=120)
    plt.close(fig)


def plot_topic_words(topics):
    k = len(topics)
    fig, axes = plt.subplots(1, k, figsize=(4 * k, 4.2))
    for ax, topic in zip(np.atleast_1d(axes), topics):
        words = topic["words"][:10]
        ax.barh([w["term"] for w in words], [w["weight"] for w in words], color="#3b6ea8")
        ax.invert_yaxis()
        ax.set_title(f"Тема {topic['topic']}")
        ax.tick_params(labelsize=9)
    fig.suptitle("NMF: топ-10 слов каждой темы", fontsize=13)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "topics_nmf_words.png", dpi=120)
    plt.close(fig)


def plot_doc_topics(names, shares):
    fig, ax = plt.subplots(figsize=(1.3 * shares.shape[1] + 4, 0.45 * len(names) + 1.6))
    im = ax.imshow(shares, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(shares.shape[1]))
    ax.set_xticklabels([f"Тема {i + 1}" for i in range(shares.shape[1])])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(shares.shape[1]):
            ax.text(j, i, f"{shares[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if shares[i, j] > 0.6 else "black")
    ax.set_title("Тематическое распределение роликов (NMF)")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "topics_doc_topic.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
