"""Features of users, books and user-book pairs for the hybrid recommender.

Everything derived from ratings is computed on the training part only. Book
metadata (Goodreads ratings_count, tags) contains no held-out answers and is used whole.
"""
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from .content import is_service

# Each rule is (token sequence, tokens that veto the match). A tag matches when the
# sequence appears in it as consecutive tokens and no veto token is present,
# so "science" counts as non-fiction but "science-fiction" does not.
GENRE_RULES = {
    'fantasy':            [(('fantasy',), ())],
    'science_fiction':    [(('science', 'fiction'), ()), (('sci', 'fi'), ()), (('scifi',), ()),
                           (('dystopia',), ()), (('dystopian',), ())],
    'romance':            [(('romance',), ()), (('romantic',), ()), (('chick', 'lit'), ())],
    'mystery_crime':      [(('mystery',), ()), (('mysteries',), ()), (('crime',), ()), (('detective',), ())],
    'thriller':           [(('thriller',), ()), (('thrillers',), ()), (('suspense',), ())],
    'horror':             [(('horror',), ())],
    'historical_fiction': [(('historical',), ())],
    'classics':           [(('classics',), ()), (('classic',), ())],
    'young_adult':        [(('young', 'adult'), ()), (('ya',), ()), (('teen',), ())],
    'children':           [(('children',), ()), (('childrens',), ()), (('kids',), ()),
                           (('picture', 'books'), ()), (('middle', 'grade'), ())],
    'comics':             [(('graphic', 'novels'), ()), (('graphic', 'novel'), ()), (('comics',), ()),
                           (('comic',), ()), (('manga',), ())],
    'nonfiction':         [(('non', 'fiction'), ()), (('nonfiction',), ()), (('biography',), ()),
                           (('memoir',), ()), (('memoirs',), ()), (('autobiography',), ()),
                           (('history',), ()), (('self', 'help'), ()), (('psychology',), ()),
                           (('philosophy',), ()), (('business',), ()), (('science',), ('fiction', 'fi'))],
    'poetry':             [(('poetry',), ()), (('poems',), ())],
    'humor':              [(('humor',), ()), (('humour',), ()), (('funny',), ()), (('comedy',), ())],
    'paranormal':         [(('paranormal',), ()), (('supernatural',), ()), (('vampire',), ()), (('vampires',), ())],
}
GENRES = list(GENRE_RULES)


def tag_tokens(tag):
    return [t for t in re.split(r'[^a-z0-9]+', str(tag).lower()) if t]


def _has_sequence(tokens, seq):
    n = len(seq)
    return any(tuple(tokens[i:i + n]) == seq for i in range(len(tokens) - n + 1))


def tag_genres(tag):
    tokens = tag_tokens(tag)
    return [genre for genre, rules in GENRE_RULES.items()
            if any(_has_sequence(tokens, seq) and not set(veto) & set(tokens) for seq, veto in rules)]


def book_genres(book_tags, index, top_tags=20):
    """Binary books x genres matrix aligned with index.books.

    A book gets a genre if any of its top_tags most frequent content tags maps to it,
    the same tags the homework content model uses.
    """
    content = book_tags[~book_tags.tag_name.map(is_service)]
    content = (content.sort_values(['book_id', 'count'], ascending=[True, False])
               .groupby('book_id').head(top_tags))
    mapping = {tag: tag_genres(tag) for tag in content.tag_name.unique()}
    pairs   = (content.assign(genre=content.tag_name.map(mapping))
               .explode('genre').dropna(subset=['genre'])[['book_id', 'genre']].drop_duplicates())
    table   = pd.crosstab(pairs.book_id, pairs.genre).clip(upper=1)
    return table.reindex(index=index.books, columns=GENRES, fill_value=0).astype(np.int8)


def genre_coverage(genres):
    """Share of books with at least one genre and the share of books per genre."""
    return float((genres.sum(axis=1) > 0).mean()), genres.mean().sort_values(ascending=False)


def user_features(train, index, like_threshold=4):
    """Activity and rating habits of each user, aligned with index.users."""
    g   = train.assign(liked=train.rating >= like_threshold).groupby('user_id')
    out = pd.DataFrame({'n_ratings':   g.size(),
                        'mean_rating': g.rating.mean(),
                        'std_rating':  g.rating.std(ddof=0),
                        'share_liked': g.liked.mean()})
    out = out.reindex(index.users)
    return out.fillna({'n_ratings': 0, 'mean_rating': train.rating.mean(),
                       'std_rating': 0, 'share_liked': (train.rating >= like_threshold).mean()})


def book_features(train, books, genres, index, m=None):
    """Popularity, smoothed quality and controversy of each book, aligned with index.books.

    log_ratings_count comes from Goodreads metadata: inside Goodbooks almost every
    book has exactly 100 ratings, so train counts cannot separate hits from niche books.
    bayes_mean uses m = median number of train ratings per book unless given.
    """
    stats = train.groupby('book_id').rating.agg(n_train='size', mean='mean', std_rating=lambda r: r.std(ddof=0))
    C     = train.rating.mean()
    m     = stats.n_train.median() if m is None else m
    stats['bayes_mean'] = (stats.n_train * stats['mean'] + m * C) / (stats.n_train + m)
    out = stats.drop(columns='mean').reindex(index.books)
    out = out.fillna({'n_train': 0, 'std_rating': 0, 'bayes_mean': C})
    out['log_ratings_count'] = np.log1p(books.set_index('id').ratings_count.reindex(index.books).fillna(0).values)
    out['n_genres'] = genres.sum(axis=1).values
    return out


class PairFeatures:
    """Similarity of a candidate book to the user's training history.

    R_ind: binary users x books training matrix. X_n: L2-normalised TF-IDF rows aligned
    with index.books. S: item-item similarity from item_cf.item_similarity on R_ind.
    genres: output of book_genres.

    content_cos equals content.user_profile_scores and cf_sum equals
    item_cf.implicit_scores at the same (user, book) cell, computed for pairs only.
    """

    def __init__(self, R_ind, X_n, S, genres):
        self.R_ind   = R_ind.tocsr()
        self.S       = S.tocsr()
        self.X_n     = X_n.tocsr()
        self.profile = normalize(normalize(self.R_ind, norm='l1') @ self.X_n).tocsr()
        G            = genres.values.astype(float)
        self.G_n     = normalize(G, norm='l1')
        self.user_g  = normalize(self.R_ind @ G, norm='l1')

    def __call__(self, u_pos, b_pos, chunk=200_000):
        u_pos, b_pos = np.asarray(u_pos), np.asarray(b_pos)
        parts = []
        for start in range(0, len(u_pos), chunk):
            u, b   = u_pos[start:start + chunk], b_pos[start:start + chunk]
            cf     = self.R_ind[u].multiply(self.S[b]).tocsr()
            parts.append(pd.DataFrame({
                'content_cos':    np.asarray(self.profile[u].multiply(self.X_n[b]).sum(axis=1)).ravel(),
                'cf_sum':         np.asarray(cf.sum(axis=1)).ravel(),
                'cf_max':         cf.max(axis=1).toarray().ravel(),
                'genre_affinity': (self.user_g[u] * self.G_n[b]).sum(axis=1),
            }))
        return pd.concat(parts, ignore_index=True)
