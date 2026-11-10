"""Non-personalised baseline: smoothed mean rating."""
import numpy as np


def popularity_scores(ratings, min_ratings=50, m=None):
    """Bayesian mean (v*R + m*C) / (v + m) for books with at least min_ratings ratings.

    m defaults to the median number of ratings per book.
    """
    stats = ratings.groupby('book_id').rating.agg(['mean', 'count'])
    C     = ratings.rating.mean()
    m     = stats['count'].median() if m is None else m
    score = (stats['count'] * stats['mean'] + m * C) / (stats['count'] + m)
    return score[stats['count'] >= min_ratings].sort_values(ascending=False)


def top_popular(ratings, n=10, exclude=(), **kwargs):
    score = popularity_scores(ratings, **kwargs)
    return score[~score.index.isin(exclude)].head(n)


def popularity_vector(scores, book_pos):
    """Align popularity scores with matrix columns; books below the threshold get -inf."""
    vec = np.full(len(book_pos), -np.inf)
    vec[book_pos.get_indexer(scores.index)] = scores.values
    return vec
