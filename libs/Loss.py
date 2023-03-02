import torch.nn.functional as F

class StandardLoss(object):

    def __init__(self, device):
        self.device = device

    def calculate_loss(self, output, target, params=None):
        self.last_lost = F.smooth_l1_loss(output, target)
        return F.smooth_l1_loss(output, target).to(self.device)
    
    def get_last_loss_strings(self):
        return str(round(float(self.last_lost), 4 )), 'NA'

class EWCLoss(object):

    def __init__(self, means, precision_matrices, importance, device):
        self.means = means
        self.precision_matrices = precision_matrices
        self.importance = importance
        self.device = device
        self.last_loss = None
        self.last_penalty = None

    def calculate_loss(self, output, target, params):

        penalty = 0
        for n, p in params:
            _loss = self.precision_matrices[n].to(self.device) * (p.to(self.device) - self.means[n].to(self.device)) ** 2
            penalty += _loss.sum()
        #print(self.importance * penalty, F.smooth_l1_loss(output, target), sum(output))

        self.last_loss = F.smooth_l1_loss(output, target)
        self.last_penalty =  self.importance * penalty

        return self.last_loss +  self.last_penalty

    def get_last_loss_strings(self):
        return str(round(float(self.last_loss), 4)), str(round(float(self.last_penalty), 4))



