"""Ranking evaluation: Precision@K, Recall@K, nDCG@K and properties of the recommended lists."""
import numpy as np
from scipy import sparse


def rmse(pred, true):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


def relevance_matrix(test, index, threshold):
    rel = test[test.rating >= threshold]
    return index.to_csr(rel, values=np.ones(len(rel)))


def top_k(scores, seen, k):
    """Top-k columns per row after masking already seen items, sorted by score."""
    rows = np.repeat(np.arange(seen.shape[0]), np.diff(seen.indptr))
    scores[rows, seen.indices] = -np.inf
    top   = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
    return np.take_along_axis(top, order, axis=1)


def evaluate(score_fn, rel, seen, item_popularity, k_list=(5, 10), chunk=2000):
    """Average ranking metrics over users with at least one relevant item.

    score_fn(users) returns a dense users x items score block. Ideal DCG is built from
    all relevant items of the user (capped at K), not from the recommended list.
    Also reports catalogue coverage of the top-K_max lists and the median popularity
    of recommended items.
    """
    k_max = max(k_list)
    users = np.flatnonzero(np.diff(rel.indptr))
    disc  = 1 / np.log2(np.arange(2, k_max + 2))
    res   = {f'{m}@{k}': [] for k in k_list for m in ('Precision', 'Recall', 'nDCG')}
    shown = []
    for start in range(0, len(users), chunk):
        u     = users[start:start + chunk]
        top   = top_k(score_fn(u), seen[u], k_max)
        r     = rel[u]
        hits  = r[np.arange(len(u))[:, None], top].toarray()
        n_rel = np.diff(r.indptr)
        for k in k_list:
            h    = hits[:, :k]
            idcg = np.cumsum(disc[:k])[np.minimum(n_rel, k) - 1]
            res[f'Precision@{k}'].append(h.sum(1) / k)
            res[f'Recall@{k}'].append(h.sum(1) / n_rel)
            res[f'nDCG@{k}'].append((h * disc[:k]).sum(1) / idcg)
        shown.append(top)
    shown = np.concatenate(shown)
    out   = {name: float(np.concatenate(v).mean()) for name, v in res.items()}
    out['покрытие каталога']     = len(np.unique(shown)) / rel.shape[1]
    out['медиана ratings_count'] = float(np.median(item_popularity[shown.ravel()]))
    return out


def ndcg_per_user(score_fn, rel, seen, k=10, chunk=2000):
    users = np.flatnonzero(np.diff(rel.indptr))
    disc  = 1 / np.log2(np.arange(2, k + 2))
    out   = []
    for start in range(0, len(users), chunk):
        u    = users[start:start + chunk]
        top  = top_k(score_fn(u), seen[u], k)
        r    = rel[u]
        hits = r[np.arange(len(u))[:, None], top].toarray()
        out.append((hits * disc).sum(1) / np.cumsum(disc)[np.minimum(np.diff(r.indptr), k) - 1])
    return users, np.concatenate(out)


def same_author_share(score_fn, rel, seen, author_codes, k=10, chunk=2000):
    """Share of top-k recommendations and of hits whose first author the user has already read."""
    users = np.flatnonzero(np.diff(rel.indptr))
    hits_same = hits_all = recs_same = recs_all = 0
    for start in range(0, len(users), chunk):
        u      = users[start:start + chunk]
        s      = seen[u]
        top    = top_k(score_fn(u), s, k)
        hits   = rel[u][np.arange(len(u))[:, None], top].toarray().astype(bool)
        rows   = np.repeat(np.arange(len(u)), np.diff(s.indptr))
        read_a = sparse.csr_matrix((np.ones(s.nnz), (rows, author_codes[s.indices])),
                                   shape=(len(u), author_codes.max() + 1))
        same   = read_a[np.arange(len(u))[:, None], author_codes[top]].toarray() > 0
        hits_same += (hits & same).sum()
        hits_all  += hits.sum()
        recs_same += same.sum()
        recs_all  += same.size
    return {'доля рекомендаций от прочитанных авторов': recs_same / recs_all,
            'доля попаданий от прочитанных авторов': hits_same / hits_all}
