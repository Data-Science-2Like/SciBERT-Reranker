import unittest

import torch

from reranker.triplet_loss import TripletLoss


class MyTestCase(unittest.TestCase):

    def test_triplet_loss_train(self):
        labels_1 = torch.tensor([1, 0, 0, 0, 0])
        preds_1 = torch.tensor([[0, 0.8], [0, 0.3], [0, 0.5], [0, 1.0], [0, 0.2]])
        triplet_loss_01 = TripletLoss(0.1)
        loss_1 = triplet_loss_01(preds_1, labels_1)
        self.assertAlmostEqual(loss_1.item(), 0.1735689476 / 4.0)

        labels_2 = torch.tensor([0, 0, 0, 1, 0])
        preds_2 = torch.tensor([[0, 1.0], [0, 0.3], [0, 0.5], [0, 0.8], [0, 0.2]])
        loss_2 = triplet_loss_01(preds_2, labels_2)
        self.assertAlmostEqual(loss_2.item(), 0.1735689476 / 4.0)

        triplet_loss_0 = TripletLoss(0.0)
        loss_0 = triplet_loss_0(preds_1, labels_1)
        self.assertAlmostEqual(loss_0.item(), 0.0410840975 / 4.0)

    def test_triplet_loss_eval(self):
        triplet_loss_01 = TripletLoss(0.1)
        triplet_loss_01.eval()

        labels_1 = torch.tensor([1, 0, 0, 0, 0])
        preds_1 = torch.tensor([[0, 0.8], [0, 0.3], [0, 0.5], [0, 1.0], [0, 0.2]])
        loss_1_eval = triplet_loss_01(preds_1, labels_1)
        self.assertAlmostEqual(loss_1_eval.item(), 0.1735689476 / 4.0)

        labels_2 = torch.tensor([0, 0, 0, 0])
        preds_2 = torch.tensor([[0, 0.3], [0, 0.5], [0, 1.0], [0, 0.2]])
        loss_2_eval = triplet_loss_01(preds_2, labels_2)
        self.assertAlmostEqual(loss_2_eval.item(), 0.1735689476 / 4.0)

        labels_3 = torch.tensor([0, 0, 0, 0, 1])
        preds_3 = torch.tensor([[0, 0.3], [0, 0.5], [0, 1.0], [0, 0.2], [0, 0.8]])
        loss_3_eval = triplet_loss_01(preds_3, labels_3)
        self.assertAlmostEqual(loss_3_eval.item(), 0.1735689476 / 4.0)

        labels_4 = torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0])
        preds_4 = torch.tensor(
            [[0, 0.3], [0, 0.5], [0, 1.0], [0, 0.2], [0, 0.8], [0, 0.3], [0, 0.5], [0, 1.0], [0, 0.2]])
        loss_4_eval = triplet_loss_01(preds_4, labels_4)
        self.assertAlmostEqual(loss_4_eval.item(), 2.0 * 0.1735689476 / 8.0)


if __name__ == '__main__':
    unittest.main()
