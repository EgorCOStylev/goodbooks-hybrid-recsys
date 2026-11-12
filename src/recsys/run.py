"""End-to-end pipeline: raw Goodbooks CSVs to evaluated hybrid and top-k recommendations.

    python run_pipeline.py --out reports/run
    python run_pipeline.py --data path/to/goodbooks --config my_config.json --out reports/run2

Stages: load and clean -> split and simulate new users -> validation (fit generators,
tune hybrid weights and MMR lambda, train the ranker) -> test (refit generators on
fit + val, evaluate all systems) -> recommend (refit on all ratings, top-k for every user).
Every stage reuses the functions the notebooks call, so the numbers match notebooks 02 and 03.
"""
import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, hybrid, metrics, pipeline, ranker

SEGMENTS = ('new', 'active')


@dataclass
class Config:
    data_dir: str = ''
    out_dir: str = 'reports/run'
    seed: int = 42
    threshold: float = 4
    k: int = 10
    n_per_model: int = 100
    n_new: int = 3000
    n_tune: int = 8000
    grid: list = field(default_factory=lambda: list(pipeline.WEIGHT_GRID))
    lambdas: list = field(default_factory=lambda: [1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    lambda_tolerance: float = 0.98
    use_ranker: bool = True
    neg_per_pos: int = 20
    epochs: int = 30
    patience: int = 3
    final_system: str = 'ranker'
    recommend: bool = True
    recommend_chunk: int = 5000

    @classmethod
    def from_json(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8')))


def log(msg, t0=[time.time()]):
    print(f'[{time.time() - t0[0]:7.0f} s] {msg}', flush=True)


def load(data_dir):
    if not data_dir:
        import kagglehub
        data_dir = kagglehub.dataset_download('zygmunt/goodbooks-10k')
    raw                     = data.load_raw(Path(data_dir))
    books                   = raw['books']
    ratings, _, _           = data.dedup_ratings(raw['ratings'])
    books['original_title'] = books.original_title.fillna(books.title)
    book_tags               = data.attach_tags(raw['book_tags'], books, raw['tags'])
    return ratings, books, book_tags


def choose_lambda(pool, score, genres, rel, cfg):
    """Smallest lambda whose nDCG@k stays within lambda_tolerance of no re-ranking."""
    ndcg = {lam: hybrid.ranking_metrics(pool, hybrid.mmr_rerank(pool, score, genres, cfg.k, lam), rel, cfg.k)[f'nDCG@{cfg.k}']
            for lam in cfg.lambdas}
    base = ndcg[1.0] if 1.0 in ndcg else hybrid.ranking_metrics(pool, score, rel, cfg.k)[f'nDCG@{cfg.k}']
    return min(lam for lam, v in ndcg.items() if v >= cfg.lambda_tolerance * base), ndcg


def train_ranker(pool, rel, users, n_ratings, cols, cfg):
    train_users, es_users = ranker.split_users(users, share=0.2, seed=cfg.seed)
    rows    = ranker.training_rows(pool, train_users, neg_per_pos=cfg.neg_per_pos, seed=cfg.seed)
    es_pool = pool[pool.user.isin(es_users)].reset_index(drop=True)
    X_es    = ranker.feature_matrix(es_pool, cols, n_ratings)
    eval_fn = lambda score_fn: hybrid.ranking_metrics(es_pool, score_fn(X_es), rel, cfg.k)[f'nDCG@{cfg.k}']
    return ranker.fit_ranker(ranker.feature_matrix(pool.iloc[rows], cols, n_ratings), pool.hit.values[rows], eval_fn,
                             epochs=cfg.epochs, patience=cfg.patience, seed=cfg.seed)


def system_scores(pool, n_ratings, weights, model, cols):
    segment = hybrid.segment_of(n_ratings)
    scores  = {'cf_implicit': hybrid.hybrid_score(pool, {'cf_implicit': 1.0}),
               'linear':      hybrid.segment_score(pool, segment, {s: weights[('features', s)] for s in SEGMENTS})}
    if model is not None:
        scores['ranker'] = ranker.predict(model, ranker.feature_matrix(pool, cols, n_ratings))
    return scores


def validation_stage(fit, val, books, book_tags, idx, cfg):
    rel             = metrics.relevance_matrix(val, idx, cfg.threshold)
    fit_t, _        = pipeline.simulate_new_users(fit, rel, idx, n_users=cfg.n_new, seed=cfg.seed)
    fitted          = pipeline.fit_models(fit_t, books, book_tags, idx)
    users           = np.flatnonzero(np.diff(rel.indptr))
    pool            = pipeline.scored_pool(fitted, users, rel, n_per_model=cfg.n_per_model)
    log(f'validation pool: {len(pool):,} candidates for {len(users):,} users')
    segment         = hybrid.segment_of(fitted.n_ratings)
    weights         = pipeline.tune_segment_weights(pool, rel, segment, users, list(fitted.generators),
                                                    grid=cfg.grid, n_tune=cfg.n_tune, k=cfg.k, seed=cfg.seed)
    log('hybrid weights tuned')
    score           = hybrid.segment_score(pool, segment, {s: weights[('features', s)] for s in SEGMENTS})
    lam, lam_curve  = choose_lambda(pool, score, fitted.genres, rel, cfg)
    log(f'MMR lambda: {lam}')
    cols            = list(fitted.generators) + list(pipeline.POOL_FEATURES)
    model, history  = (None, None)
    if cfg.use_ranker:
        model, history = train_ranker(pool, rel, users, fitted.n_ratings, cols, cfg)
        log(f'ranker trained for {len(history)} epochs')
    return {'weights': weights, 'lambda': lam, 'lambda_curve': lam_curve, 'model': model, 'history': history, 'cols': cols}


def test_stage(fit, val, test, books, book_tags, idx, art, cfg):
    rel        = metrics.relevance_matrix(test, idx, cfg.threshold)
    train_t, _ = pipeline.simulate_new_users(pd.concat([fit, val], ignore_index=True), rel, idx, n_users=cfg.n_new, seed=cfg.seed)
    fitted     = pipeline.fit_models(train_t, books, book_tags, idx)
    users      = np.flatnonzero(np.diff(rel.indptr))
    pool       = pipeline.scored_pool(fitted, users, rel, n_per_model=cfg.n_per_model)
    segment    = hybrid.segment_of(fitted.n_ratings)
    scores     = system_scores(pool, fitted.n_ratings, art['weights'], art['model'], art['cols'])
    for name in list(scores):
        if name != 'cf_implicit':
            scores[f'{name} + MMR'] = hybrid.mmr_rerank(pool, scores[name], fitted.genres, cfg.k, art['lambda'])
    pop        = books.set_index('id').ratings_count.reindex(idx.books).to_numpy()
    groups     = {'все': users, 'новые': users[segment[users] == 'new'], 'активные': users[segment[users] == 'active']}
    n          = fitted.n_ratings
    for label, lo, hi in (('4–10', 4, 10), ('11–30', 11, 30), ('31–100', 31, 100), ('>100', 101, 10**9)):
        groups[f'активные {label}'] = users[(segment[users] == 'active') & (n[users] >= lo) & (n[users] <= hi)]
    rows = []
    for name, score in scores.items():
        row = {'система': name}
        for label, u in groups.items():
            row[f'nDCG@{cfg.k}, {label}'] = hybrid.ranking_metrics(pool, score, rel, cfg.k, users=u)[f'nDCG@{cfg.k}']
        m = hybrid.ranking_metrics(pool, score, rel, cfg.k)
        row.update({f'Precision@{cfg.k}': m[f'Precision@{cfg.k}'], f'Recall@{cfg.k}': m[f'Recall@{cfg.k}'],
                    **hybrid.list_properties(pool, score, fitted.genres, pop, len(idx.books), cfg.k)})
        rows.append(row)
    recall = {label: hybrid.pool_recall(pool[pool.user.isin(groups[label])], rel)['pool'] for label in ('новые', 'активные')}
    return pd.DataFrame(rows), recall


def recommend_stage(ratings, books, book_tags, idx, art, cfg):
    """Fit on every rating and write the final top-k for all users, chunk by chunk."""
    fitted = pipeline.fit_models(ratings, books, book_tags, idx)
    users  = np.arange(len(idx.users))
    out    = []
    for start in range(0, len(users), cfg.recommend_chunk):
        u     = users[start:start + cfg.recommend_chunk]
        pool  = pipeline.scored_pool(fitted, u, n_per_model=cfg.n_per_model)
        score = system_scores(pool, fitted.n_ratings, art['weights'], art['model'], art['cols'])[cfg.final_system]
        final = hybrid.mmr_rerank(pool, score, fitted.genres, cfg.k, art['lambda'])
        rows, pos = hybrid.topk_rows(pool.user.values, final, cfg.k)
        out.append(pd.DataFrame({'user_id': idx.users[pool.user.values[rows]], 'rank': pos + 1,
                                 'book_id': idx.books[pool.book.values[rows]]}))
        log(f'recommended for {min(start + cfg.recommend_chunk, len(users)):,} / {len(users):,} users')
    recs = pd.concat(out, ignore_index=True)
    return recs.merge(books[['id', 'title', 'authors']].rename(columns={'id': 'book_id'}), on='book_id', how='left')


def run_pipeline(cfg):
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding='utf-8')
    if cfg.final_system == 'ranker' and not cfg.use_ranker:
        raise ValueError('final_system="ranker" needs use_ranker=true')

    ratings, books, book_tags = load(cfg.data_dir)
    log(f'data: {len(ratings):,} ratings, {ratings.user_id.nunique():,} users, {len(books):,} books')
    idx            = data.Index(users=np.sort(ratings.user_id.unique()), books=np.sort(books.id.unique()))
    fit, val, test = data.split_train_val_test(ratings, seed=cfg.seed)

    art = validation_stage(fit, val, books, book_tags, idx, cfg)
    weights = {f'{v}, {s}': w for (v, s), w in art['weights'].items()}
    (out / 'weights.json').write_text(json.dumps(weights, ensure_ascii=False, indent=2), encoding='utf-8')
    if art['history'] is not None:
        art['history'].to_csv(out / 'ranker_history.csv', index=False)

    table, recall = test_stage(fit, val, test, books, book_tags, idx, art, cfg)
    table.to_csv(out / 'metrics_test.csv', index=False)
    log('test metrics written')

    summary = {'ratings': len(ratings), 'users': int(ratings.user_id.nunique()), 'books': len(books),
               'mmr_lambda': art['lambda'], 'mmr_curve_ndcg': art['lambda_curve'],
               'pool_recall_test': recall, 'final_system': cfg.final_system}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding='utf-8')

    if cfg.recommend:
        recs = recommend_stage(ratings, books, book_tags, idx, art, cfg)
        recs.to_csv(out / 'recommendations.csv', index=False)
        log(f'recommendations written: {len(recs):,} rows')
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description='Hybrid book recommender: from raw CSVs to metrics and recommendations.')
    parser.add_argument('--data', default='', help='folder with Goodbooks-10k CSVs; downloaded with kagglehub if omitted')
    parser.add_argument('--out', default=None, help='output folder (default reports/run)')
    parser.add_argument('--config', default=None, help='JSON file overriding Config fields')
    parser.add_argument('--no-ranker', action='store_true', help='skip the neural ranker (no torch needed); recommendations then come from the linear hybrid')
    parser.add_argument('--no-recommend', action='store_true', help='stop after the test evaluation')
    args = parser.parse_args(argv)

    cfg = Config.from_json(args.config) if args.config else Config()
    if args.data:
        cfg.data_dir = args.data
    if args.out:
        cfg.out_dir = args.out
    if args.no_ranker:
        cfg.use_ranker, cfg.final_system = False, 'linear'
    if args.no_recommend:
        cfg.recommend = False
    table = run_pipeline(cfg)
    with pd.option_context('display.width', 200, 'display.max_columns', 30):
        print(table.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
