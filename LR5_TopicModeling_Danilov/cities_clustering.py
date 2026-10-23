"""
Пример 2: кластерный анализ городов по блюдам.

Каждый город кодируется вектором нормализованной частотности блюд:
    norm_freq(город, блюдо) = (число сайтов города с блюдом) / (число сайтов города)
после чего города кластеризуются тремя методами (k-means, иерархическая,
спектральная), а число кластеров подбирается методом локтя и silhouette.

Запуск:
    python cities_clustering.py
    python cities_clustering.py --k 3 --max-share 0.6
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "data" / "dishes_raw.json"
RESULTS_DIR = BASE_DIR / "results"
RANDOM_STATE = 42

# Синонимы и варианты написания -> каноническое название блюда.
# Одно упоминание может дать несколько блюд: «пельмени с лосятиной» -> пельмени + лосятина.
CANON_RULES = [
    (r"эчпочмак|учпочмак", "эчпочмак"),
    (r"бешбармак|бишбармак", "бешбармак"),
    (r"б[эе]л[иеэ]ш", "бэлиш"),
    (r"токмач|тукмас", "токмач"),
    (r"кыстыбый", "кыстыбый"),
    (r"чак-чак", "чак-чак"),
    (r"\bказы", "казы"),
    (r"кумыс", "кумыс"),
    (r"катык", "катык"),
    (r"посикунч|посекунч", "посикунчики"),
    (r"вязовск|кунгурск\w* \w*\s*пряник|печатный пряник", "вязовские пряники"),
    (r"пельмен", "пельмени"),
    (r"калитк", "калитки"),
    (r"лохикейт", "лохикейто"),
    (r"корюшк", "корюшка"),
    (r"пышк", "пышки"),
    (r"\bуха\b", "уха"),
    (r"\bборщ", "борщ"),
    (r"\bщи\b", "щи"),
    (r"шаньг", "шаньги"),
    (r"расстега", "расстегаи"),
    (r"калач", "калачи"),
    (r"пирож(к|ок)", "пирожки"),
    (r"\bпиро(г|ги)\b|кулебяк|курник|рыбник", "пироги"),
    (r"\bблин", "блины"),
    (r"рассольник", "рассольник"),
    (r"солянк", "солянка"),
    (r"строганин", "строганина"),
    (r"олен", "оленина"),
    (r"лос(ят|я\b|ь)", "лосятина"),
    (r"медвеж", "медвежатина"),
    (r"кабан", "кабан"),
    (r"косул", "косуля"),
    (r"конин", "конина"),
    (r"баран", "баранина"),
    (r"\bгус", "гусь"),
    (r"морошк", "морошка"),
    (r"брусни", "брусника"),
    (r"клюкв", "клюква"),
    (r"черник|черничн", "черника"),
    (r"кедров", "кедровый орех"),
    (r"\bикр", "икра"),
    (r"лосос|\bкета\b|горбуш|нерк|кижуч|чавыч|красная рыба", "лососевые"),
    (r"\bкраб", "краб"),
    (r"гребеш", "гребешок"),
    (r"папоротник", "папоротник"),
    (r"лимонник", "лимонник"),
    (r"иван-чай|копорск", "иван-чай"),
    (r"ряпушк", "ряпушка"),
    (r"\bсиг", "сиг"),
    (r"форел", "форель"),
    (r"щук", "щука"),
    (r"налим", "налим"),
    (r"муксун", "муксун"),
    (r"омул", "омуль"),
    (r"нельм", "нельма"),
    (r"стерля", "стерлядь"),
    (r"мид(и|ии)", "мидии"),
    (r"гриб|грузд|волнушк|моховик|лисичк|подосиновик", "грибы"),
    (r"\bмед\b|\bмёд|медов", "мёд"),
    (r"варень|конфитюр", "варенье"),
    (r"кофе|\bраф\b|латте", "кофе"),
    (r"\bчай\b", "чай"),
    (r"квас", "квас"),
    (r"морс", "морс"),
    (r"сбитень", "сбитень"),
    (r"чебурек", "чебуреки"),
    (r"беляш", "беляши"),
    (r"\bманты", "манты"),
    (r"бууз", "буузы"),
    (r"\bплов", "плов"),
    (r"шаверм|шаурм", "шаверма"),
    (r"пицц", "пицца"),
    (r"бургер", "бургер"),
    (r"хот-дог|сосиск", "хот-дог"),
    (r"мороженое", "мороженое"),
    (r"\bторт", "торты"),
    (r"пирожн", "пирожные"),
    (r"шашлык", "шашлык"),
    (r"сырник", "сырники"),
    (r"\bкаш", "каша"),
    (r"\bсур\b", "сур"),
    (r"пончик|плюшк|булочк|бублик|баранк", "сдоба"),
    (r"пряник", "пряники"),
    (r"глинтвейн", "глинтвейн"),
    (r"пиво", "пиво"),
    (r"\bводк", "водка"),
    (r"настойк|наливк|бальзам", "настойки"),
    (r"толокно", "толокно"),
    (r"черемш", "черемша"),
]


def canonical_dishes(raw):
    text = raw.lower().replace("«", "").replace("»", "").replace('"', "").strip()
    found = {canon for pattern, canon in CANON_RULES if re.search(pattern, text)}
    if "вязовские пряники" in found:
        found.discard("пряники")
    return found if found else {text}


def load(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cities = list(data["cities"])
    site_sets = {}
    for city in cities:
        site_sets[city] = [set().union(*[canonical_dishes(d) for d in site["dishes"]]) for site in data["cities"][city]]
    return cities, site_sets


def build_matrix(cities, site_sets, max_share, min_cities):
    dish_cities = Counter()
    for city in cities:
        for dish in set().union(*site_sets[city]):
            dish_cities[dish] += 1

    n = len(cities)
    common = sorted(d for d, c in dish_cities.items() if c / n > max_share)
    dishes = sorted(d for d, c in dish_cities.items() if c >= min_cities and c / n <= max_share)

    X = np.zeros((n, len(dishes)))
    for i, city in enumerate(cities):
        sites = site_sets[city]
        for j, dish in enumerate(dishes):
            X[i, j] = sum(dish in s for s in sites) / len(sites)
    return X, dishes, common, dish_cities


def choose_k(Xn, k_min, k_max):
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(Xn)
        sil = silhouette_score(Xn, km.labels_, metric="cosine")
        rows.append((k, float(km.inertia_), float(sil)))
    best = max(rows, key=lambda r: r[2])[0]
    return best, rows


def cluster(Xn, k):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(Xn)
    hier = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Xn)
    spec = SpectralClustering(n_clusters=k, affinity="cosine", random_state=RANDOM_STATE).fit_predict(Xn)
    return {"kmeans": km, "hier": hier, "spectral": spec}


def relabel(reference, labels):
    """Приводит номера кластеров к нумерации reference, чтобы цвета совпадали."""
    mapping, used = {}, set()
    for lab in sorted(set(labels), key=lambda l: -np.sum(labels == l)):
        votes = Counter(reference[labels == lab])
        for ref_lab, _ in votes.most_common():
            if ref_lab not in used:
                mapping[lab] = ref_lab
                used.add(ref_lab)
                break
        else:
            mapping[lab] = max(set(reference) | used) + 1
            used.add(mapping[lab])
    return np.array([mapping[l] for l in labels])


def plot_k_selection(rows, best, path):
    ks, inertia, sil = zip(*rows)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(ks, inertia, marker="o", color="#3b6ea8")
    ax1.set_title("Метод локтя (k-means)")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia")
    ax1.grid(alpha=0.3)
    ax2.plot(ks, sil, marker="o", color="#c0504d")
    ax2.axvline(best, ls="--", color="gray")
    ax2.set_title(f"Silhouette (косинусная метрика), лучшее k={best}")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Silhouette score")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_pca(Xn, cities, labels, path):
    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(Xn)
    titles = {"kmeans": "k-means", "hier": "Иерархическая (ward)", "spectral": "Спектральная (cosine)"}
    palette = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for ax, (name, lab) in zip(axes, labels.items()):
        for c in sorted(set(lab)):
            idx = lab == c
            ax.scatter(coords[idx, 0], coords[idx, 1], s=160, color=palette(c), label=f"Кластер {c + 1}",
                       edgecolor="black", linewidth=0.6)
        for i, ((x, y), city) in enumerate(zip(coords, cities)):
            ax.annotate(city, (x, y), textcoords="offset points", xytext=(7, 6 if i % 2 == 0 else -15), fontsize=10)
        ax.set_title(titles[name])
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Города в пространстве блюд (PCA, 2 компоненты)", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_dendrogram(Xn, cities, path):
    Z = linkage(Xn, method="ward")
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(Z, labels=cities, leaf_rotation=30, ax=ax, color_threshold=0.7 * Z[:, 2].max())
    ax.set_title("Дендрограмма городов (ward, L2-нормированные векторы)")
    ax.set_ylabel("Расстояние")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_similarity(Xn, cities, path):
    sim = Xn @ Xn.T
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(sim, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(cities)))
    ax.set_yticks(range(len(cities)))
    ax.set_xticklabels(cities, rotation=40, ha="right")
    ax.set_yticklabels(cities)
    for i in range(len(cities)):
        for j in range(len(cities)):
            ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if sim[i, j] > 0.6 else "black")
    ax.set_title("Косинусная близость городов по блюдам")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_city_dish(X, cities, dishes, order, path, top=30):
    weight = X.sum(axis=0)
    top_idx = np.argsort(-weight)[:top]
    sub = X[np.ix_(order, top_idx)]
    fig, ax = plt.subplots(figsize=(15, 5.5))
    im = ax.imshow(sub, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([cities[i] for i in order])
    ax.set_xticks(range(len(top_idx)))
    ax.set_xticklabels([dishes[i] for i in top_idx], rotation=60, ha="right", fontsize=9)
    ax.set_title(f"Нормализованная частотность {top} самых распространённых блюд (города упорядочены по кластерам)")
    fig.colorbar(im, ax=ax, label="norm_freq")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Кластеризация городов по блюдам")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--k", default="auto", help="число кластеров или auto")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument("--max-share", type=float, default=0.6,
                        help="исключить блюда, встречающиеся более чем в этой доле городов")
    parser.add_argument("--min-cities", type=int, default=1)
    parser.add_argument("--output", default="cities")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    prefix = RESULTS_DIR / args.output

    cities, site_sets = load(args.input)
    X, dishes, common, dish_cities = build_matrix(cities, site_sets, args.max_share, args.min_cities)
    Xn = normalize(X)

    print(f"Городов: {len(cities)}, сайтов: {sum(len(s) for s in site_sets.values())}, "
          f"блюд в словаре: {len(dishes)}")
    print(f"Исключены общие блюда (> {args.max_share:.0%} городов): {', '.join(common) or 'нет'}")

    rows = None
    if args.k == "auto":
        best_k, rows = choose_k(Xn, args.k_min, min(args.k_max, len(cities) - 1))
        print("\nПодбор k:")
        for k, inertia, sil in rows:
            print(f"  k={k}  inertia={inertia:.3f}  silhouette={sil:.3f}")
        plot_k_selection(rows, best_k, f"{prefix}_k_selection.png")
        k = best_k
    else:
        k = int(args.k)

    labels = cluster(Xn, k)
    labels["hier"] = relabel(labels["kmeans"], labels["hier"])
    labels["spectral"] = relabel(labels["kmeans"], labels["spectral"])

    ari = {
        "kmeans_vs_hier": adjusted_rand_score(labels["kmeans"], labels["hier"]),
        "kmeans_vs_spectral": adjusted_rand_score(labels["kmeans"], labels["spectral"]),
        "hier_vs_spectral": adjusted_rand_score(labels["hier"], labels["spectral"]),
    }

    print(f"\nk = {k}")
    print(f"  {'город':<17}{'kmeans':>8}{'hier':>7}{'spectral':>10}")
    for i, city in enumerate(cities):
        print(f"  {city:<17}{labels['kmeans'][i] + 1:>8}{labels['hier'][i] + 1:>7}{labels['spectral'][i] + 1:>10}")
    print("\nСогласованность методов (ARI): " + ", ".join(f"{a}={v:.2f}" for a, v in ari.items()))

    clusters = []
    for c in sorted(set(labels["kmeans"])):
        idx = np.where(labels["kmeans"] == c)[0]
        mean = X[idx].mean(axis=0)
        top = np.argsort(-mean)[:8]
        clusters.append({
            "cluster": int(c) + 1,
            "cities": [cities[i] for i in idx],
            "top_dishes": [{"dish": dishes[j], "mean_norm_freq": round(float(mean[j]), 3)} for j in top if mean[j] > 0],
        })
        print(f"  Кластер {c + 1}: {', '.join(cities[i] for i in idx)}")
        print(f"    топ-блюда: {', '.join(dishes[j] for j in top if mean[j] > 0)}")

    order = list(np.lexsort((np.arange(len(cities)), labels["kmeans"])))
    plot_pca(Xn, cities, labels, f"{prefix}_clusters_pca.png")
    plot_dendrogram(Xn, cities, f"{prefix}_dendrogram.png")
    plot_similarity(Xn, cities, f"{prefix}_similarity.png")
    plot_city_dish(X, cities, dishes, order, f"{prefix}_city_dish_heatmap.png")

    with open(f"{prefix}_matrix.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["город"] + dishes)
        for city, row in zip(cities, X):
            writer.writerow([city] + [round(float(v), 3) for v in row])

    with open(f"{prefix}_labels.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["city", "kmeans", "hier", "spectral"])
        for i, city in enumerate(cities):
            writer.writerow([city] + [int(labels[m][i]) + 1 for m in ("kmeans", "hier", "spectral")])

    shared = defaultdict(list)
    for dish, c in dish_cities.items():
        shared[c].append(dish)

    report = {
        "cities": cities,
        "sites_per_city": {c: len(site_sets[c]) for c in cities},
        "n_dishes": len(dishes),
        "excluded_common_dishes": common,
        "k": k,
        "k_selection": [{"k": r[0], "inertia": round(r[1], 4), "silhouette": round(r[2], 4)} for r in rows] if rows else None,
        "silhouette": {m: round(float(silhouette_score(Xn, l, metric="cosine")), 4) for m, l in labels.items()},
        "ari": {a: round(float(v), 4) for a, v in ari.items()},
        "clusters": clusters,
    }
    Path(f"{prefix}_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nРезультаты сохранены в results/ с префиксом '{args.output}'")


if __name__ == "__main__":
    main()
