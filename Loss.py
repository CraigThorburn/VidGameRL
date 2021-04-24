import torch.nn.functional as F

class StandardLoss(object):

    def __init__(self, device):
        pass

    def calculate_loss(self, output, target, params=None):
        return F.smooth_l1_loss(output, target).to(device)

class EWCLoss(object):

    def __init__(self, means, precision_matrices, importance, device):
        self.means = means
        self.precision_matrices = precision_matrices
        self.importance = importance

    def calculate_loss(self, output, target, params):

        penalty = 0
        for n, p in params:
            _loss = self.precision_matrices[n] * (p - self.means[n]) ** 2
            penalty += _loss.sum()
        print(self.importance * penalty, F.smooth_l1_loss(output, target), sum(output))
        return F.smooth_l1_loss(output, target) + self.importance * penalty


