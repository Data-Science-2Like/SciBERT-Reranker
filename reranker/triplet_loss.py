import torch


# triplet loss definition from https://arxiv.org/abs/2112.01206

# architecture/structure based on loss implementations of simpletransformers
# (https://github.com/ThilinaRajapakse/simpletransformers/tree/master/simpletransformers/losses)


class TripletLoss(torch.nn.Module):
    r"""Criterion that computes the triplet loss for the SciBERT-based Reranker.
    According to https://arxiv.org/abs/2112.01206, we compute the loss as follows:
    .. math::
        \mathcal{L}(q, d_-, d_+) = \max[ s(q,d_-) - s(q,d_+) + m, 0 ]
    where:
       - :math:`q` is the query
       - :math:`d_+` is the positive / searched document
       - :math:`d_-` is a negative / not searched document
       - :math:`s(a,b)` is the sigmoid of the network output for a and b for the relevant class
       - :math:`m>0` is the margin, e.g. m = 0.1
    Shape:
        - Input: :math:`(N,2)` where the second entries are the logit output of the network for being relevant
        - Target: :math:`(N,)` where all values are zero despite the one for the positive document :math:`d_+` (one-hot coded)
    """

    def __init__(self, m: float = 0.1) -> None:
        super(TripletLoss, self).__init__()
        self.m: float = m

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:

        if len(input.shape) == 2:
            if input.shape[0] != target.shape[0]:
                raise ValueError("Number of elements in input and target shapes must be the same. Got: {}"
                                 .format(input.shape, input.shape))
            if input.shape[1] != 2:
                raise ValueError("Invalid input shape, we expect (N,2). Got (N,{})"
                                 .format(input.shape[1]))
        else:
            raise ValueError("Invalid input shape, we expect (N,2). Got: {}"
                             .format(input.shape))
        if torch.sum(target) != 1:
            raise ValueError(
                "Invalid target tensor, we expect the target to be one-hot coded. Got: sum over target = {}"
                .format(torch.sum(target)))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}".format(
                    input.device, target.device))

        # compute sigmoid in order to retrieve score s()
        input = input[:, 1]  # select label for being relevant
        input_score = torch.sigmoid(input)

        # gain positive score and negative score tensor
        positive_idx = torch.nonzero(target).item()
        positive_score = input_score[positive_idx]
        negative_scores = torch.cat([input_score[:positive_idx], input_score[positive_idx + 1:]])

        # compute loss
        loss = positive_score - negative_scores + self.m
        loss[loss < 0] = 0  # maximum operation: max[loss, 0]

        return torch.mean(loss)
