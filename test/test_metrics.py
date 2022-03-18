import unittest

import numpy as np

from reranker.metrics import MeanRecallAtK, MeanReciprocalRank, _check_shape, _rank_labels


class MyTestCase(unittest.TestCase):

    def test_ranked_labels(self):
        labels = np.asarray([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        preds = np.asarray([[0, 0.8], [0, 0.3], [0, 0.5], [0, 1.0], [0, 0.2],
                            [0, 0.4], [0, 0.3], [0, 0.5], [0, 1.0], [0, 0.2],
                            [0, 0.1], [0, 0.3], [0, 0.5], [0, 1.0], [0, 0.2]])

        labels_per_query, preds_per_query = _check_shape(labels, preds, 5)
        ranked_labels = _rank_labels(labels_per_query, preds_per_query)
        self.assertEqual(ranked_labels.tolist(), [[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 1]])

    def test_mean_recall_at_k(self):
        labels = np.asarray([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        preds = np.asarray([[0, 0.8], [0, 0.3], [0, 0.5], [0, 1.0], [0, 0.2],
                            [0, 0.4], [0, 0.3], [0, 0.5], [0, 1.0], [0, 0.2],
                            [0, 0.1], [0, 0.3], [0, 0.5], [0, 1.0], [0, 0.2]])

        recall_7 = MeanRecallAtK(5, 7)
        result = recall_7(labels, preds)
        self.assertEqual(result, 1)
        recall_5 = MeanRecallAtK(5, 5)
        result = recall_5(labels, preds)
        self.assertEqual(result, 1)
        recall_4 = MeanRecallAtK(5, 4)
        result_4 = recall_4(labels, preds)
        self.assertEqual(result_4, 2 / 3)
        recall_3 = MeanRecallAtK(5, 3)
        result_3 = recall_3(labels, preds)
        self.assertEqual(result_3, 2 / 3)
        recall_2 = MeanRecallAtK(5, 2)
        result_2 = recall_2(labels, preds)
        self.assertEqual(result_2, 1 / 3)
        recall_1 = MeanRecallAtK(5, 1)
        result_1 = recall_1(labels, preds)
        self.assertEqual(result_1, 0)

    def test_mean_reciprocal_rank(self):
        labels = np.asarray([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        preds = np.asarray([[0, 0.8], [0, 0.3], [0, 0.5], [0, 1.0], [0, 0.2],
                            [0, 0.4], [0, 0.3], [0, 0.5], [0, 1.0], [0, 0.2],
                            [0, 0.1], [0, 0.3], [0, 0.5], [0, 1.0], [0, 0.2]])

        mrr = MeanReciprocalRank(5)
        result = mrr(labels, preds)
        self.assertEqual(result, 1 / 3 * (1 / 2 + 1 / 3 + 1 / 5))


if __name__ == '__main__':
    unittest.main()
