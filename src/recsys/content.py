"""Content-based model on book titles and user tags."""
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# Tags made only of these words describe a reader's shelf, not the book.
SERVICE_WORDS = {
    'to', 'read', 'reread', 're', 'currently', 'reading', 'in', 'own', 'owned', 'i', 'it', 'have',
    'favorites', 'favourites', 'favorite', 'favourite', 'fav', 'faves', 'all', 'time',
    'default', 'library', 'kindle', 'ebook', 'ebooks', 'e', 'book', 'books', 'audio',
    'audiobook', 'audiobooks', 'audible', 'wish', 'list', 'wishlist', 'maybe', 'borrowed',
    'abandoned', 'dnf', 'did', 'didn', 't', 'not', 'finish', 'finished', 'unfinished',
    'reviewed', 'shelfari', 'goodreads', 'buy', 'bought', 'tbr', 'my', 'mine', 'gave', 'up',
    'paperback', 'hardcover', 'nook', 'calibre', 'physical', 'bookshelf', 'shelf', 'home',
    'club', 'english', 'series', 'novel', 'novels', 'star', 'stars', 'five',
}


def is_service(tag):
    words = [w for w in re.split(r'[^a-z0-9]+', str(tag).lower()) if w]
    return bool(words) and all(w in SERVICE_WORDS or w.isdigit() for w in words)


def build_profiles(books, book_tags, top_tags=20):
    """Text profile per book: title words plus its top content tags, one token per tag."""
    content = book_tags[~book_tags.tag_name.map(is_service)]
    content = (content.sort_values(['book_id', 'count'], ascending=[True, False])
               .groupby('book_id').head(top_tags))
    tokens  = (content.assign(token=content.tag_name.str.lower().str.replace(r'[^a-z0-9]+', '_', regex=True).str.strip('_'))
               .groupby('book_id').token.apply(' '.join))
    titles  = books.set_index('id').original_title.str.lower().str.replace(r'[^\w\s]', ' ', regex=True)
    return (titles + ' ' + tokens.reindex(titles.index).fillna('')).str.strip()


def fit_tfidf(profiles):
    vectorizer = TfidfVectorizer(token_pattern=r'[^\s]+', stop_words='english', min_df=2, sublinear_tf=True)
    return vectorizer, vectorizer.fit_transform(profiles.values)


def similar_books(X, book_ids, book_id, n=5):
    """Top-n books by cosine similarity of L2-normalised TF-IDF rows."""
    i    = int(np.flatnonzero(book_ids == book_id)[0])
    sims = (X[i] @ X.T).toarray().ravel()
    sims[i] = -1
    top  = np.argpartition(-sims, n)[:n]
    top  = top[np.argsort(-sims[top])]
    return pd.Series(sims[top], index=book_ids[top], name='similarity')


def user_profile_scores(R_ind_block, X_n):
    """Cosine between each book and the mean TF-IDF vector of the user's read books."""
    profile = normalize(normalize(R_ind_block, norm='l1') @ X_n)
    return (profile @ X_n.T).toarray()
