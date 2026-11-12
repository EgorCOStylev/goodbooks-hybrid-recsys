"""Neural re-ranker over the hybrid candidate pool.

A small MLP scores every (user, book) candidate from its pool columns (generator rank
scores and feature percentiles) plus the user's history length. A linear weighted sum
cannot use the history length: it is the same for all candidates of a user and does not
change their order. Inside an MLP it can switch which signals matter for short and long
histories.

torch is imported inside the functions so the data helpers work without it.
"""
import copy

import numpy as np
import pandas as pd

USER_COLUMN = 'log_n_ratings'


def feature_matrix(pool, cols, n_ratings, use_user=True, max_ratings=200):
    """float32 matrix of pool columns, plus log history length scaled to about [0, 1]."""
    X = pool[cols].to_numpy(np.float32)
    if use_user:
        log_n = np.log1p(n_ratings[pool.user.values]) / np.log1p(max_ratings)
        X = np.column_stack([X, log_n.astype(np.float32)])
    return np.ascontiguousarray(X)


def training_rows(pool, users, neg_per_pos=20, seed=42):
    """Rows of the given users: every relevant candidate and neg_per_pos random others per relevant one."""
    rows = np.flatnonzero(pool.user.isin(users).values)
    hit  = pool.hit.values[rows].astype(bool)
    pos, neg = rows[hit], rows[~hit]
    rng  = np.random.default_rng(seed)
    neg  = rng.choice(neg, min(len(neg), neg_per_pos * len(pos)), replace=False)
    return rng.permutation(np.concatenate([pos, neg]))


def split_users(users, share=0.2, seed=42):
    """Random split of users into ranker training users and early-stopping users."""
    users = np.random.default_rng(seed).permutation(users)
    n     = int(len(users) * share)
    return np.sort(users[n:]), np.sort(users[:n])


def build_mlp(n_in, hidden=(64, 32), dropout=0.1):
    import torch.nn as nn
    layers, d = [], n_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


def predict(model, X, batch=65536):
    import torch
    device = next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(model(torch.from_numpy(X[i:i + batch]).to(device)).squeeze(-1).cpu().numpy())
    return np.concatenate(out)


def fit_ranker(X, y, eval_fn, hidden=(64, 32), dropout=0.1, epochs=10, batch=4096, lr=1e-3,
               weight_decay=1e-5, patience=2, seed=42):
    """Train the MLP with binary cross-entropy and keep the epoch with the best eval_fn.

    eval_fn(score_fn) receives a function mapping a feature matrix to scores and returns
    the early-stopping metric (nDCG@10 on held-out users of the validation pool).
    Training stops after `patience` epochs without improvement.
    """
    import torch
    torch.manual_seed(seed)
    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    model   = build_mlp(X.shape[1], hidden, dropout).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    Xt, yt  = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))
    best, best_state, bad, history = -np.inf, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        perm, total = torch.randperm(len(Xt)), 0.0
        for i in range(0, len(Xt), batch):
            idx    = perm[i:i + batch]
            xb, yb = Xt[idx].to(device), yt[idx].to(device)
            loss   = loss_fn(model(xb).squeeze(-1), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        metric = eval_fn(lambda Z: predict(model, Z))
        history.append({'эпоха': epoch, 'loss': total / len(Xt), 'nDCG@10 (ранняя остановка)': metric})
        if metric > best + 1e-5:
            best, best_state, bad = metric, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)
