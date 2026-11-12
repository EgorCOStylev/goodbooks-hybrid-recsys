"""End-to-end steps of the hybrid recommender: fit every candidate generator on a
training frame, simulate new users, build the scored candidate pool."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from . import content, features, hybrid, item_cf, popularity
from .data import truncate_history

GENERATORS    = ('cf_implicit', 'cf_explicit', 'content', 'svd', 'known', 'bayes')
POOL_FEATURES = ('content_cos', 'cf_sum', 'cf_max', 'genre_affinity', 'bayes_mean', 'log_ratings_count', 'std_rating')
SVD_PARAMS    = {'n_factors': 50, 'reg_all': 0.05, 'n_epochs': 20, 'lr_all': 0.005, 'random_state': 42}
WEIGHT_GRID   = [-0.5, 0, 0.05, 0.1, 0.25, 0.5, 1, 2]


@dataclass
class Fitted:
    index: object
    R_ind: object
    generators: dict
    pair: features.PairFeatures
    book_feats: pd.DataFrame
    genres: pd.DataFrame

    @property
    def n_ratings(self):
        return np.diff(self.R_ind.indptr)


def fit_models(train, books, book_tags, index, use=GENERATORS, svd_params=SVD_PARAMS,
               k_neighbors=50, shrink=10, extra_generators=None):
    """Fit the homework models on train and wrap each as score_fn(user_positions).

    known ranks by Goodreads ratings_count (log), bayes is the homework popularity
    model (smoothed mean rating, -inf below its count threshold).
    extra_generators maps a name to factory(train, index, R_ind) -> score_fn.
    """
    R                       = index.to_csr(train)
    R_dev, R_ind, user_mean = item_cf.center_by_user(R)
    S_imp                   = item_cf.item_similarity(R_ind, R_ind, k_neighbors, shrink)
    _, X                    = content.fit_tfidf(content.build_profiles(books, book_tags).reindex(index.books).fillna(''))
    X_n                     = normalize(X)
    known                   = np.log1p(books.set_index('id').ratings_count.reindex(index.books).fillna(0).to_numpy(float))
    gens = {}
    if 'cf_implicit' in use:
        gens['cf_implicit'] = lambda u: item_cf.implicit_scores(R_ind, S_imp, u)
    if 'cf_explicit' in use:
        S_exp = item_cf.item_similarity(R_dev, R_ind, k_neighbors, shrink)
        gens['cf_explicit'] = lambda u: item_cf.explicit_scores(R_dev, R_ind, user_mean, S_exp, u)
    if 'content' in use:
        gens['content'] = lambda u: content.user_profile_scores(R_ind[u], X_n)
    if 'svd' in use:
        from surprise import SVD
        from .svd import SVDScorer, to_surprise
        _, trainset, _ = to_surprise(train, train.head(0))
        gens['svd'] = SVDScorer(SVD(**svd_params).fit(trainset), trainset, index)
    if 'known' in use:
        gens['known'] = lambda u: np.tile(known, (len(u), 1))
    if 'bayes' in use:
        bayes = popularity.popularity_vector(popularity.popularity_scores(train), index.book_pos)
        gens['bayes'] = lambda u: np.tile(bayes, (len(u), 1))
    for name, factory in (extra_generators or {}).items():
        gens[name] = factory(train, index, R_ind)
    genres = features.book_genres(book_tags, index)
    return Fitted(index=index, R_ind=R_ind, generators=gens,
                  pair=features.PairFeatures(R_ind, X_n, S_imp, genres),
                  book_feats=features.book_features(train, books, genres, index),
                  genres=genres)


def simulate_new_users(train, rel, index, n_users=3000, keep=(1, 2, 3), min_train=10, seed=42):
    """Cut the train history of randomly chosen evaluable users down to 1-3 books.

    Only users with at least min_train train ratings and a relevant held-out book are
    eligible, so each simulated new user still has something to be evaluated on.
    The chosen users are split evenly between the keep values. Returns the truncated
    train frame and the ids of the simulated new users.
    """
    counts   = train.groupby('user_id').size()
    has_rel  = index.users[np.flatnonzero(np.diff(rel.indptr))]
    eligible = np.intersect1d(counts.index[counts >= min_train], has_rel)
    chosen   = np.random.default_rng(seed).choice(eligible, min(n_users, len(eligible)), replace=False)
    for group, k in zip(np.array_split(chosen, len(keep)), keep):
        train = truncate_history(train, group, k, seed=seed)
    return train, np.sort(chosen)


def scored_pool(fitted, users, rel=None, n_per_model=100, chunk=2000, feature_cols=POOL_FEATURES):
    """Candidate pool with rank-normalised feature columns and, if rel is given, relevance labels."""
    pool = hybrid.build_pool(fitted.generators, users, fitted.R_ind, n_per_model, chunk)
    if rel is not None:
        pool = hybrid.attach_relevance(pool, rel)
    cols = (fitted.pair(pool.user.values, pool.book.values)
            .join(fitted.book_feats.iloc[pool.book.values].reset_index(drop=True)))
    pool = hybrid.add_columns(pool, cols[list(feature_cols)])
    return hybrid.rank_within_user(pool, list(feature_cols))


def tune_segment_weights(pool, rel, segment, users, generators, grid=WEIGHT_GRID, n_tune=8000, k=10, seed=42):
    """Staged coordinate ascent per segment, the procedure of notebook 02.

    For each segment the generator weights are tuned first (start: cf_implicit = 1),
    then the pool features are added at zero weight and tuning continues. Active users
    are tuned on a random sample of n_tune users. Returns {(variant, segment): weights}
    with variant 'models' or 'features'.
    """
    rng        = np.random.default_rng(seed)
    seg_users  = {s: users[segment[users] == s] for s in ('new', 'active')}
    tune_users = {'new': seg_users['new'],
                  'active': rng.choice(seg_users['active'], min(n_tune, len(seg_users['active'])), replace=False)}
    init_models = {g: float(g == 'cf_implicit') for g in generators}
    weights = {}
    for s, u in tune_users.items():
        weights[('models', s)], _, _ = hybrid.tune_weights(pool, rel, init_models, grid, k=k, users=u)
        init = {**weights[('models', s)], **{f: 0.0 for f in POOL_FEATURES}}
        weights[('features', s)], _, _ = hybrid.tune_weights(pool, rel, init, grid, k=k, users=u)
    return weights
