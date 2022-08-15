import numpy as np


# information retrieval metrics for usage with simpletransformers Classification model
# (https://github.com/ThilinaRajapakse/simpletransformers)


class MeanReciprocalRank:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __call__(self, labels, preds):
        r"""Method that computes the mean reciprocal rank metric (for our use case).
        Shape:
            - preds: :math:`(N,2)` where the second entries are the logit output of the network for being relevant
            - labels: :math:`(N,)` where all values are zero despite the one for the positive document :math:`d_+` (one-hot coded)
        """
        labels_per_query, preds_per_query = _check_shape(labels, preds, self.batch_size)
        ranked_labels = _rank_labels(labels_per_query, preds_per_query)

        first_relevant_pos = np.argmax(ranked_labels, axis=1) + 1
        reciprocal_rank_per_query = 1.0 / first_relevant_pos
        no_pos_doc_idx = np.where(~np.any(ranked_labels, axis=1))
        reciprocal_rank_per_query[no_pos_doc_idx] = 0

        return np.mean(reciprocal_rank_per_query), np.std(reciprocal_rank_per_query)


class MeanRecallAtK:
    def __init__(self, batch_size, k):
        self.batch_size = batch_size
        self.k = k

    def __call__(self, labels, preds):
        r"""Method that computes the recall at k metric (for our use case).
        Shape:
            - preds: :math:`(N,2)` where the second entries are the logit output of the network for being relevant
            - labels: :math:`(N,)` where all values are zero despite the one for the positive document :math:`d_+` (one-hot coded)
            - k: integer indicating the cutoff point
        """
        labels_per_query, preds_per_query = _check_shape(labels, preds, self.batch_size)
        ranked_labels = _rank_labels(labels_per_query, preds_per_query)

        ranked_labels_cutoff = ranked_labels[:, :self.k]
        recall_at_k_per_query = np.sum(ranked_labels_cutoff, axis=1) / np.sum(labels_per_query, axis=1)
        return np.mean(recall_at_k_per_query), np.std(recall_at_k_per_query)


def _rank_labels(labels, preds):
    # compute sigmoid in order to retrieve scores
    preds_relevant = preds[:, :, 1]  # select label for being relevant
    ranking_scores = _sigmoid(preds_relevant)

    # rank the labels according to the ranking scores
    descending_idx = np.argsort(-ranking_scores)
    return labels[np.arange(descending_idx.shape[0])[:, None], descending_idx]


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _check_shape(labels, preds, batch_size):
    if len(preds.shape) == 2:
        if preds.shape[0] != labels.shape[0]:
            raise ValueError("Number of elements in preds and labels shapes must be the same. Got: {}"
                             .format(preds.shape, preds.shape))
        if preds.shape[0] % batch_size != 0:
            raise ValueError("Number of elements in preds must be dividable by batch_size. Got: {}"
                             .format(preds.shape, batch_size))
        if preds.shape[1] != 2:
            raise ValueError("Invalid preds shape, we expect (N,2). Got (N,{})"
                             .format(preds.shape[1]))
    else:
        raise ValueError("Invalid preds shape, we expect (N,2). Got: {}"
                         .format(preds.shape))

    labels_reshaped = np.reshape(labels, (-1, batch_size))
    preds_reshaped = np.reshape(preds, (-1, batch_size, 2))
    return labels_reshaped, preds_reshaped
