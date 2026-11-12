"""Hybrid recommender: candidate pool from several models, rank-normalised scores,
weighted sum per user segment and genre-aware re-ranking (MMR)."""
import numpy as np
import pandas as pd

from .metrics import top_k


def build_pool(generators, users, seen, n_per_model=100, chunk=2000):
    """Union of every generator's top-n books per user, without books seen in train.

    generators maps a name to score_fn(user_positions) -> dense users x books block.
    Each generator column holds the rank score 1 - rank / n (1 for the model's first
    book, 0 when the book is outside its top-n), so models with different score scales
    can be summed. Books a generator scores -inf (e.g. below a popularity threshold)
    are not taken from it.
    """
    names, n_items = list(generators), seen.shape[1]
    parts = []
    for start in range(0, len(users), chunk):
        u = np.asarray(users[start:start + chunk])
        keys, model, value = [], [], []
        for j, name in enumerate(names):
            s     = np.array(generators[name](u), dtype=float)
            top   = top_k(s, seen[u], n_per_model)
            valid = np.isfinite(np.take_along_axis(s, top, axis=1))
            rank  = np.broadcast_to(np.arange(n_per_model), top.shape)
            rows  = np.broadcast_to(np.arange(len(u))[:, None], top.shape)
            keys.append((rows * n_items + top)[valid])
            value.append(1 - rank[valid] / n_per_model)
            model.append(np.full(valid.sum(), j))
        uniq, inv = np.unique(np.concatenate(keys), return_inverse=True)
        M = np.zeros((len(uniq), len(names)), dtype=np.float32)
        M[inv, np.concatenate(model)] = np.concatenate(value)
        part = pd.DataFrame(M, columns=names)
        part.insert(0, 'book', (uniq % n_items).astype(np.int32))
        part.insert(0, 'user', u[uniq // n_items].astype(np.int32))
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def attach_relevance(pool, rel):
    """Add a hit column: 1 if the pooled book is relevant for the user in rel."""
    pool['hit'] = np.asarray(rel[pool.user.values, pool.book.values]).ravel().astype(np.int8)
    return pool


def pool_recall(pool, rel, generators=None):
    """Share of each user's relevant books that made it into the pool (and into each generator's top-n).

    This is the ceiling for any ranking done on the pool.
    """
    n_rel  = np.diff(rel.indptr)
    users  = np.intersect1d(np.unique(pool.user), np.flatnonzero(n_rel))
    sub    = pool[pool.user.isin(users)]
    out    = {'pool': _mean_share(sub.user.values, sub.hit.values, n_rel, users)}
    for name in generators or []:
        mask = sub[name].values > 0
        out[name] = _mean_share(sub.user.values[mask], sub.hit.values[mask], n_rel, users)
    return out


def _mean_share(user, hit, n_rel, users):
    found = np.bincount(user, weights=hit, minlength=len(n_rel))
    return float((found[users] / n_rel[users]).mean())


def add_columns(pool, frame):
    """Append feature columns computed row-wise for the pool (same order as pool)."""
    for col in frame.columns:
        pool[col] = frame[col].to_numpy(np.float32)
    return pool


def rank_within_user(pool, cols):
    """Replace feature columns by their percentile rank among the user's candidates.

    Puts features with different units (log counts, mean ratings, cosines) on the
    same [0, 1] scale as the generator rank scores.
    """
    pool[cols] = pool.groupby('user')[cols].rank(pct=True).astype(np.float32)
    return pool


def hybrid_score(pool, weights):
    cols = list(weights)
    return pool[cols].to_numpy(np.float32) @ np.array([weights[c] for c in cols], dtype=np.float32)


def topk_rows(user, score, k):
    """Pool rows of each user's top-k by score, with their positions 0..k-1."""
    order    = np.lexsort((-score, user))
    u_sorted = user[order]
    start    = np.r_[0, np.flatnonzero(np.diff(u_sorted)) + 1]
    pos      = np.arange(len(order)) - np.repeat(start, np.diff(np.r_[start, len(order)]))
    keep     = pos < k
    return order[keep], pos[keep]


def ranking_metrics(pool, score, rel, k=10, users=None):
    """Precision@k, Recall@k and nDCG@k of the top-k taken from the pool.

    Same definitions as metrics.evaluate: averaged over users with at least one
    relevant book, ideal DCG built from all relevant books of the user (capped at k),
    so relevant books missing from the pool count as misses.
    """
    n_rel = np.diff(rel.indptr)
    users = np.unique(pool.user) if users is None else np.asarray(users)
    users = users[n_rel[users] > 0]
    if len(users) == 0:
        return {f'Precision@{k}': np.nan, f'Recall@{k}': np.nan, f'nDCG@{k}': np.nan}
    rows, pos = topk_rows(pool.user.values, score, k)
    hit   = pool.hit.values[rows].astype(float)
    disc  = 1 / np.log2(np.arange(2, k + 2))
    u     = pool.user.values[rows]
    hits  = np.bincount(u, weights=hit, minlength=len(n_rel))[users]
    dcg   = np.bincount(u, weights=hit * disc[pos], minlength=len(n_rel))[users]
    idcg  = np.cumsum(disc)[np.minimum(n_rel[users], k) - 1]
    return {f'Precision@{k}': float((hits / k).mean()),
            f'Recall@{k}':    float((hits / n_rel[users]).mean()),
            f'nDCG@{k}':      float((dcg / idcg).mean())}


def tune_weights(pool, rel, init, grid, k=10, n_pass=2, users=None):
    """Coordinate ascent on nDCG@k: change one weight at a time over the grid.

    A full grid over ten weights is infeasible; coordinate ascent finds a local
    optimum and its history shows how much each column contributes.
    """
    if users is not None:
        pool = pool[pool.user.isin(users)].reset_index(drop=True)
    cols  = list(init)
    X     = pool[cols].to_numpy(np.float32)
    w     = np.array([init[c] for c in cols], dtype=np.float32)
    key   = f'nDCG@{k}'
    best  = ranking_metrics(pool, X @ w, rel, k)[key]
    history = [('init', None, best)]
    for _ in range(n_pass):
        for j, col in enumerate(cols):
            for v in grid:
                if v == w[j]:
                    continue
                trial    = w.copy()
                trial[j] = v
                if not trial.any():
                    continue
                score = ranking_metrics(pool, X @ trial, rel, k)[key]
                if score > best + 1e-6:
                    best, w = score, trial
                    history.append((col, float(v), best))
    return dict(zip(cols, w.tolist())), best, pd.DataFrame(history, columns=['column', 'weight', key])


def segment_of(n_ratings, new_max=3):
    return np.where(np.asarray(n_ratings) <= new_max, 'new', 'active')


def segment_score(pool, user_segment, weights_by_segment):
    """Hybrid score where each user's candidates use the weights of the user's segment."""
    seg   = user_segment[pool.user.values]
    score = np.zeros(len(pool), dtype=np.float32)
    for name, weights in weights_by_segment.items():
        mask        = seg == name
        score[mask] = hybrid_score(pool[mask], weights)
    return score


def genre_jaccard(G):
    inter = G @ G.T
    size  = G.sum(axis=1)
    union = size[:, None] + size[None, :] - inter
    return np.divide(inter, union, out=np.zeros(inter.shape), where=union > 0)


def mmr_rerank(pool, score, genres, k=10, lam=0.7, top_m=50):
    """Greedy MMR on each user's top_m candidates.

    Next book maximises lam * score - (1 - lam) * max Jaccard(genres, already chosen),
    score min-max scaled within the user's top_m so lam means the same for everyone.
    Returns a score vector where the chosen books get k..1 in order and the rest -inf,
    so it plugs into ranking_metrics directly.
    """
    G         = genres.to_numpy(dtype=float)
    rows, pos = topk_rows(pool.user.values, score, top_m)
    users     = pool.user.values[rows]
    bounds    = np.r_[0, np.flatnonzero(np.diff(users)) + 1, len(rows)]
    out       = np.full(len(pool), -np.inf, dtype=np.float32)
    books     = pool.book.values
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx  = rows[a:b]
        s    = score[idx].astype(float)
        span = s.max() - s.min()
        s    = (s - s.min()) / span if span > 0 else np.ones_like(s)
        sim  = genre_jaccard(G[books[idx]])
        chosen, max_sim = [0], sim[0].copy()
        for _ in range(min(k, len(idx)) - 1):
            val         = lam * s - (1 - lam) * max_sim
            val[chosen] = -np.inf
            j           = int(np.argmax(val))
            chosen.append(j)
            max_sim     = np.maximum(max_sim, sim[j])
        out[idx[chosen]] = np.arange(len(chosen), 0, -1)
    return out


def list_properties(pool, score, genres, book_popularity, n_items, k=10):
    """Diversity and popularity of the final top-k lists.

    ILS: mean pairwise genre Jaccard inside a list (lower is more diverse).
    Distinct genres: genres covered by a list. Coverage: share of the catalogue
    recommended to anyone. Median popularity: median of book_popularity over all slots.
    """
    G         = genres.to_numpy(dtype=float)
    rows, _   = topk_rows(pool.user.values, score, k)
    users     = pool.user.values[rows]
    books     = pool.book.values[rows]
    bounds    = np.r_[0, np.flatnonzero(np.diff(users)) + 1, len(rows)]
    ils, n_genres = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        g   = G[books[a:b]]
        sim = genre_jaccard(g)
        m   = len(g)
        ils.append((sim.sum() - np.trace(sim)) / (m * (m - 1)) if m > 1 else 0.0)
        n_genres.append(int((g.sum(axis=0) > 0).sum()))
    return {'ILS (жанры)':          float(np.mean(ils)),
            'жанров в списке':      float(np.mean(n_genres)),
            'покрытие каталога':    len(np.unique(books)) / n_items,
            'медиана популярности': float(np.median(book_popularity[books]))}
