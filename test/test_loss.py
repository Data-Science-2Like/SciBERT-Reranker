import unittest

import torch

from reranker.triplet_loss import TripletLoss


class MyTestCase(unittest.TestCase):

    def test_triplet_loss(self):
        labels_1 = torch.tensor([1, 0, 0, 0, 0])
        preds_1 = torch.tensor([[0, 0.8], [0, 0.3], [0, 0.5], [0, 1.0], [0, 0.2]])
        triplet_loss_01 = TripletLoss(0.1)
        loss_1 = triplet_loss_01(preds_1, labels_1)
        self.assertAlmostEqual(loss_1.item(), 0.1705258751)

        labels_2 = torch.tensor([0, 0, 0, 1, 0])
        preds_2 = torch.tensor([[0, 1.0], [0, 0.3], [0, 0.5], [0, 0.8], [0, 0.2]])
        loss_2 = triplet_loss_01(preds_2, labels_2)
        self.assertAlmostEqual(loss_2.item(), 0.1705258751)

        triplet_loss_0 = TripletLoss(0.0)
        loss_0 = triplet_loss_0(preds_1, labels_1)
        self.assertAlmostEqual(loss_0.item(), 0.08079689951)


if __name__ == '__main__':
    unittest.main()
