"""Glue between surprise's SVD and the matrix layout used for evaluation."""
import numpy as np


def to_surprise(train, test):
    from surprise import Dataset, Reader
    data     = Dataset.load_from_df(train[['user_id', 'book_id', 'rating']], Reader(rating_scale=(1, 5)))
    trainset = data.build_full_trainset()
    testset  = list(test[['user_id', 'book_id', 'rating']].itertuples(index=False, name=None))
    return data, trainset, testset


class SVDScorer:
    """Scores mu + b_u + b_i + q_i . p_u for all books at once, aligned with matrix columns."""

    def __init__(self, model, trainset, index):
        self.model    = model
        self.mu       = trainset.global_mean
        item_pos      = index.book_pos.get_indexer([trainset.to_raw_iid(i) for i in range(trainset.n_items)])
        self.inner_u  = np.array([trainset.to_inner_uid(u) for u in index.users])
        self.bi       = np.full(len(index.books), -np.inf)
        self.bi[item_pos] = model.bi
        self.qi       = np.zeros((len(index.books), model.qi.shape[1]))
        self.qi[item_pos] = model.qi

    def __call__(self, users):
        iu = self.inner_u[users]
        return self.mu + self.model.bu[iu][:, None] + self.bi[None, :] + self.model.pu[iu] @ self.qi.T
