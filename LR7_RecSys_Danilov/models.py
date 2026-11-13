# модели рекомендаций: популярное, контентная TF-IDF, SVD (surprise), ALS и BPR (implicit)

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from data import RELEVANT_RATING


class Recommender:
    name = "base"

    def fit(self, train, movies, tags):
        self.items = movies["item"].to_numpy()
        self.item_index = {item: i for i, item in enumerate(self.items)}
        self.seen = train.groupby("user")["item"].apply(set).to_dict()
        counts = train.groupby("item").size()
        self.popularity = np.zeros(len(self.items))
        for item, n in counts.items():
            self.popularity[self.item_index[item]] = n
        return self

    def scores(self, user):
        raise NotImplementedError

    def recommend(self, user, k):
        scores = self.scores(user).astype(float)
        seen = [self.item_index[i] for i in self.seen.get(user, ())]
        scores[seen] = -np.inf
        top = np.argpartition(-scores, k)[:k]
        top = top[np.argsort(-scores[top])]
        return [int(self.items[i]) for i in top if np.isfinite(scores[i])]

    def explain(self, user, item):
        return ""


class Popular(Recommender):
    """Бейзлайн: самые часто оцениваемые фильмы в train, одинаковые для всех."""
    name = "popular"

    def scores(self, user):
        return self.popularity


class ContentTfidf(Recommender):
    """Фильм описываю жанрами, тегами пользователей и десятилетием, профиль - взвешенное среднее понравившегося."""
    name = "content"

    def __init__(self, max_features=10000, min_df=2):
        self.max_features = max_features
        self.min_df = min_df

    @staticmethod
    def documents(movies, tags):
        tag_text = (tags.assign(tag=tags["tag"].astype(str).str.lower().str.replace(r"[^\w]+", "_", regex=True))
                    .groupby("item")["tag"].apply(lambda t: " ".join(t)))
        docs = []
        for row in movies.itertuples():
            genres = " ".join(f"genre_{g.lower().replace('-', '_').replace(' ', '_')}" for g in row.genre_list)
            decade = f"decade_{int(row.year) // 10 * 10}" if not np.isnan(row.year) else ""
            title = row.title.rsplit("(", 1)[0]
            docs.append(" ".join([genres, genres, decade, tag_text.get(row.item, ""), title]))
        return docs

    def fit(self, train, movies, tags):
        super().fit(train, movies, tags)
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, min_df=self.min_df,
                                          token_pattern=r"(?u)\b\w\w+\b", sublinear_tf=True)
        self.X = normalize(self.vectorizer.fit_transform(self.documents(movies, tags)))
        self.terms = self.vectorizer.get_feature_names_out()
        mean = train.groupby("user")["rating"].transform("mean")
        train = train.assign(weight=(train["rating"] - mean).clip(lower=0))
        train.loc[train["rating"] >= RELEVANT_RATING, "weight"] += 0.5
        self.profiles = {}
        for user, group in train.groupby("user"):
            group = group[group["weight"] > 0]
            if group.empty:
                continue
            rows = [self.item_index[i] for i in group["item"]]
            w = group["weight"].to_numpy()
            profile = np.asarray(self.X[rows].T @ w).ravel() / w.sum()
            self.profiles[user] = profile / (np.linalg.norm(profile) or 1)
        self.pop_tiebreak = self.popularity / (self.popularity.max() or 1) * 1e-3
        return self

    def scores(self, user):
        profile = self.profiles.get(user)
        if profile is None:
            return self.popularity
        return np.asarray(self.X @ profile).ravel() + self.pop_tiebreak

    def explain(self, user, item):
        profile = self.profiles.get(user)
        if profile is None:
            return "популярное"
        row = self.X[self.item_index[item]].toarray().ravel()
        top = np.argsort(-(row * profile))[:3]
        return ", ".join(self.terms[i].replace("genre_", "") for i in top if row[i] * profile[i] > 0)


class SurpriseSVD(Recommender):
    """SVD из surprise на явных оценках, ранжирую все фильмы по предсказанной оценке."""
    name = "svd"

    def __init__(self, factors=50, epochs=20, lr=0.005, reg=0.02, seed=42):
        self.params = dict(n_factors=factors, n_epochs=epochs, lr_all=lr, reg_all=reg, random_state=seed)

    def fit(self, train, movies, tags):
        from surprise import SVD, Dataset, Reader

        super().fit(train, movies, tags)
        data = Dataset.load_from_df(train[["user", "item", "rating"]], Reader(rating_scale=(0.5, 5.0)))
        self.trainset = data.build_full_trainset()
        self.algo = SVD(**self.params)
        self.algo.fit(self.trainset)
        known = [self.trainset.to_raw_iid(i) for i in range(self.trainset.n_items)]
        self.catalog_rows = np.array([self.item_index[i] for i in known])
        return self

    def scores(self, user):
        scores = np.full(len(self.items), -np.inf)
        try:
            u = self.trainset.to_inner_uid(user)
        except ValueError:
            return self.popularity
        a = self.algo
        est = self.trainset.global_mean + a.bu[u] + a.bi + a.qi @ a.pu[u]
        scores[self.catalog_rows] = est
        return scores

    def rating_metrics(self, test):
        from surprise import accuracy

        triples = list(test[["user", "item", "rating"]].itertuples(index=False, name=None))
        preds = self.algo.test(triples)
        known = [p for p in preds if not p.details.get("was_impossible", False)]
        return {"rmse": accuracy.rmse(preds, verbose=False), "mae": accuracy.mae(preds, verbose=False),
                "predictions": len(preds), "fully_known": len(known)}


class ImplicitModel(Recommender):
    """Обёртка над implicit: оценки превращаю в неявные взаимодействия «пользователь смотрел фильм»."""

    def build_matrix(self, train):
        self.users = np.sort(train["user"].unique())
        self.user_index = {u: i for i, u in enumerate(self.users)}
        rows = train["user"].map(self.user_index).to_numpy()
        cols = train["item"].map(self.item_index).to_numpy()
        values = self.confidence(train["rating"].to_numpy())
        return sp.csr_matrix((values, (rows, cols)), shape=(len(self.users), len(self.items)), dtype=np.float32)

    def confidence(self, ratings):
        return 1.0 + self.alpha * np.clip(ratings - 2.5, 0, None)

    def fit(self, train, movies, tags):
        super().fit(train, movies, tags)
        self.user_items = self.build_matrix(train)
        self.model.fit(self.user_items, show_progress=False)
        return self

    def recommend(self, user, k):
        if user not in self.user_index:
            return Recommender.recommend(self, user, k)
        u = self.user_index[user]
        ids, _ = self.model.recommend(u, self.user_items[u], N=k, filter_already_liked_items=True)
        return [int(self.items[i]) for i in ids]

    def scores(self, user):
        return self.popularity


class ImplicitALS(ImplicitModel):
    name = "als"

    def __init__(self, factors=64, regularization=0.05, alpha=2.0, iterations=20, seed=42):
        from implicit.als import AlternatingLeastSquares

        self.alpha = alpha
        self.model = AlternatingLeastSquares(factors=factors, regularization=regularization,
                                             iterations=iterations, random_state=seed, use_gpu=False)

    def explain(self, user, item):
        if user not in self.user_index:
            return "популярное"
        u = self.user_index[user]
        contributions = None
        for matrix in (self.user_items[u], self.user_items):
            try:
                _, contributions, _ = self.model.explain(u, matrix, self.item_index[item], N=1)
                break
            except Exception:
                continue
        if not contributions:
            return ""
        return int(self.items[contributions[0][0]])


class ImplicitBPR(ImplicitModel):
    name = "bpr"

    def __init__(self, factors=64, learning_rate=0.01, regularization=0.01, iterations=100, seed=42):
        from implicit.bpr import BayesianPersonalizedRanking

        self.alpha = 0.0
        self.model = BayesianPersonalizedRanking(factors=factors, learning_rate=learning_rate,
                                                 regularization=regularization, iterations=iterations,
                                                 random_state=seed, use_gpu=False)


MODELS = {
    "popular": Popular,
    "content": ContentTfidf,
    "svd": SurpriseSVD,
    "als": ImplicitALS,
    "bpr": ImplicitBPR,
}
