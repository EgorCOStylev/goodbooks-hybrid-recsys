"""Loading, cleaning and splitting the Goodbooks-10k data (first version, 981,756 ratings)."""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


def load_raw(data_dir):
    data_dir = Path(data_dir)
    return {name: pd.read_csv(data_dir / f'{name}.csv') for name in ('ratings', 'books', 'tags', 'book_tags')}


def dedup_ratings(ratings):
    """Collapse repeated user-book pairs into one row with the mean rating.

    Returns the cleaned frame, the number of rows in repeated pairs and the number
    of pairs whose repeated ratings disagree.
    """
    dup_rows  = ratings.duplicated(['user_id', 'book_id'], keep=False)
    conflicts = ratings[dup_rows].groupby(['user_id', 'book_id']).rating.nunique().gt(1).sum()
    cleaned   = ratings.groupby(['user_id', 'book_id'], as_index=False).rating.mean()
    return cleaned, int(dup_rows.sum()), int(conflicts)


def attach_tags(book_tags, books, tags):
    """Map book_tags from goodreads_book_id to books.id and add tag names.

    ratings.csv refers to books by books.id, book_tags.csv by goodreads_book_id.
    """
    out = (book_tags
           .merge(books[['id', 'book_id']], left_on='goodreads_book_id', right_on='book_id')
           .drop(columns=['goodreads_book_id', 'book_id'])
           .rename(columns={'id': 'book_id'})
           .merge(tags, on='tag_id'))
    out['count'] = out['count'].clip(lower=0)
    return out


def split_per_user(ratings, test_share=0.2, min_user_ratings=5, seed=42):
    """Random hold-out inside each user.

    Users with fewer than min_user_ratings ratings stay entirely in train:
    they still help to estimate item similarities but cannot be evaluated.
    """
    rng       = np.random.default_rng(seed)
    n_user    = ratings.groupby('user_id').user_id.transform('size')
    rank      = ratings.assign(rnd=rng.random(len(ratings))).groupby('user_id').rnd.rank(method='first')
    test_mask = (n_user >= min_user_ratings) & (rank <= np.floor(n_user * test_share))
    return ratings[~test_mask].reset_index(drop=True), ratings[test_mask].reset_index(drop=True)


@dataclass
class Index:
    """Positions of users and books in the interaction matrix."""
    users: np.ndarray
    books: np.ndarray
    user_pos: pd.Index = field(init=False)
    book_pos: pd.Index = field(init=False)

    def __post_init__(self):
        self.user_pos = pd.Index(self.users)
        self.book_pos = pd.Index(self.books)

    @property
    def shape(self):
        return len(self.users), len(self.books)

    def to_csr(self, df, values=None):
        values = df.rating.values if values is None else values
        return sparse.csr_matrix((values, (self.user_pos.get_indexer(df.user_id), self.book_pos.get_indexer(df.book_id))),
                                 shape=self.shape)
