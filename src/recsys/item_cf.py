"""Item-based collaborative filtering on explicit and implicit feedback."""
import numpy as np
from scipy import sparse


def center_by_user(R):
    """Return user-centred ratings, the binary read matrix and user means."""
    n_rated   = np.diff(R.indptr)
    user_mean = np.asarray(R.sum(axis=1)).ravel() / np.maximum(n_rated, 1)
    R_dev     = R.copy()
    R_dev.data = R_dev.data - np.repeat(user_mean, n_rated)
    R_ind     = R.copy()
    R_ind.data = np.ones_like(R_ind.data)
    return R_dev, R_ind, user_mean


def item_similarity(C, B, k_neighbors=50, shrink=10, chunk=1000):
    """Cosine between item columns of C, shrunk by co-raters: sim * n / (n + shrink).

    C is users x items (centred ratings for adjusted cosine, binary for implicit),
    B is the binary users x items matrix used to count common raters. Only the
    k_neighbors most similar items with positive similarity are kept per item.
    Computed in blocks of items so that only chunk x n_items values are dense at once.
    """
    C, B  = C.T.tocsr(), B.T.tocsr()
    norms = np.sqrt(np.asarray(C.multiply(C).sum(axis=1)).ravel())
    n     = C.shape[0]
    rows, cols, vals = [], [], []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        dot  = (C[start:stop] @ C.T).toarray()
        co   = (B[start:stop] @ B.T).toarray()
        sim  = dot / np.outer(norms[start:stop], norms).clip(1e-12) * co / (co + shrink)
        sim[np.arange(stop - start), np.arange(start, stop)] = 0
        top  = np.argpartition(-sim, k_neighbors, axis=1)[:, :k_neighbors]
        val  = np.take_along_axis(sim, top, axis=1)
        keep = val > 0
        rows.append(np.repeat(np.arange(start, stop), k_neighbors)[keep.ravel()])
        cols.append(top[keep])
        vals.append(val[keep])
    return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))


def explicit_scores(R_dev, R_ind, user_mean, S, users, shrink=1.0):
    """Predicted ratings for a block of users over all books.

    r = mean_u + sum s_ij * (r_uj - mean_u) / (sum s_ij + shrink); the extra term
    pulls the prediction to the user's mean when the neighbour support is weak.
    """
    num = (R_dev[users] @ S.T).toarray()
    den = (R_ind[users] @ S.T).toarray()
    return user_mean[users][:, None] + num / (den + shrink)


def implicit_scores(R_ind, S, users):
    """Sum of similarities between each book and the books the user has read."""
    return (R_ind[users] @ S.T).toarray()


def predict_pairs(R_dev, R_ind, user_mean, S, u_pos, b_pos, shrink=1.0, chunk=4000):
    """Predictions and neighbour support for (user, book) pairs, clipped to [1, 5]."""
    pred, support = np.empty(len(u_pos)), np.empty(len(u_pos))
    for start in range(0, R_dev.shape[0], chunk):
        mask = (u_pos >= start) & (u_pos < start + chunk)
        if not mask.any():
            continue
        block = np.arange(start, min(start + chunk, R_dev.shape[0]))
        num   = (R_dev[block] @ S.T).toarray()
        den   = (R_ind[block] @ S.T).toarray()
        uu, bb = u_pos[mask] - start, b_pos[mask]
        pred[mask]    = user_mean[u_pos[mask]] + num[uu, bb] / (den[uu, bb] + shrink)
        support[mask] = den[uu, bb]
    return np.clip(pred, 1, 5), support
