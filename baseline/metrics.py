import numpy as np


class Metric:
    def __init__(self, name):
        self.name = name

    def __call__(self, ranked_ids, cited_ids):
        raise NotImplementedError

    @staticmethod
    def create_ranked_labels(ranked_ids, cited_ids):
        ranked_labels = np.zeros_like(ranked_ids, dtype=float)
        for i, sample in enumerate(cited_ids):
            one_idx = np.where(np.isin(ranked_ids[i], sample))
            for idx in one_idx:
                ranked_labels[i][idx] = 1
        return ranked_labels


class MeanReciprocalRank(Metric):
    def __init__(self, name="MRR"):
        super().__init__(name)

    def __call__(self, ranked_ids, cited_ids):
        ranked_labels = self.create_ranked_labels(ranked_ids, cited_ids)
        first_relevant_pos = np.argmax(ranked_labels, axis=1) + 1
        reciprocal_rank_per_query = 1.0 / first_relevant_pos
        no_cited_in_ranked_idx = np.where(~np.any(ranked_labels, axis=1))
        reciprocal_rank_per_query[no_cited_in_ranked_idx] = 0
        return np.mean(reciprocal_rank_per_query), np.std(reciprocal_rank_per_query)


class MeanRecallAtK(Metric):
    def __init__(self, k, name=None):
        super().__init__(name if name is not None else "r@" + str(k))
        self.k = k

    def __call__(self, ranked_ids, cited_ids):
        ranked_labels = self.create_ranked_labels(ranked_ids, cited_ids)
        ranked_labels_cutoff = ranked_labels[:, :self.k]
        labels_per_query = [len(sample) for sample in cited_ids]
        recall_at_k_per_query = np.sum(ranked_labels_cutoff, axis=1) / labels_per_query
        return np.mean(recall_at_k_per_query), np.std(recall_at_k_per_query)
