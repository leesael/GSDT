import math

import torch
from torch import nn
import numpy as np


def count_nodes(depth, num_branches):
    """
    Count the number of nodes in the current depth.
    """
    return int(num_branches ** depth)


class Leaf(nn.Module):
    """
    Leaf class that learns static distributions for the target classes.
    """

    def __init__(self, num_nodes, num_branches):
        """
        Class initializer.
        """
        super().__init__()
        self.logit = nn.Parameter(torch.randn(num_nodes, num_branches), requires_grad=True)
        self.num_nodes = num_nodes
        self.num_branches = num_branches

    def forward(self, x):
        """
        Return the learned logits for the current input.
        """
        return self.logit.expand((x.size(0), self.num_nodes, self.num_branches))


def woodbury_logdet(diag_a, mat_u, rank):
    """
    Apply the matrix determinant lemma of Equation (7) in the paper.
    """
    mat_a = torch.diag_embed(diag_a, dim1=1, dim2=2)
    mat_a_inv = torch.diag_embed(1 / diag_a, dim1=1, dim2=2)
    mat_ut = mat_u.transpose(1, 2)
    mat_i = torch.eye(rank, rank, device=diag_a.device).unsqueeze(0).repeat(diag_a.size(0), 1, 1)
    out1 = torch.logdet(mat_i + mat_ut.matmul(mat_a_inv).matmul(mat_u))
    return out1 + torch.logdet(mat_a)


def woodbury_inverse(diag_a, mat_u, rank):
    """
    Apply the Woodbury matrix identity of Equation (8) in the paper.
    """
    mat_a_inv = torch.diag_embed(1 / diag_a, dim1=1, dim2=2)
    mat_ut = mat_u.transpose(1, 2)
    mat_i = torch.eye(rank, rank, device=diag_a.device).unsqueeze(0).repeat(diag_a.size(0), 1, 1)
    out1 = torch.inverse(mat_i + mat_ut.matmul(mat_a_inv).matmul(mat_u))
    out2 = mat_a_inv.matmul(mat_u).matmul(out1).matmul(mat_ut).matmul(mat_a_inv)
    return mat_a_inv - out2


class Layer(nn.Module):
    """
    Layer class that models all intermediate nodes as Gaussian mixtures.
    """

    def __init__(self, in_features, num_nodes, num_branches, cov, rank):
        """
        Class initializer.
        """
        super().__init__()
        self.in_features = in_features
        self.cov = cov
        self.rank = rank
        self.num_nodes = num_nodes
        self.num_branches = num_branches
        self.softmax = nn.Softmax(dim=2)

        self.weight = nn.Parameter(torch.zeros(num_branches * num_nodes, in_features), requires_grad=True)

        assert cov in ['identity', 'diagonal', 'lowrank', 'both']
        if cov == 'diagonal':
            self.cov_di = nn.Parameter(torch.ones(num_branches * num_nodes, in_features), requires_grad=True)
            self.cov_di.data *= np.log(np.e - 1)
            self.softplus = nn.Softplus()
        elif cov == 'lowrank':
            self.cov_lr = nn.Parameter(torch.ones(num_branches * num_nodes, in_features, rank), requires_grad=True)
            self.cov_lr.data /= math.sqrt(in_features * rank)
        elif cov == 'both':
            self.cov_di = nn.Parameter(torch.ones(num_branches * num_nodes, in_features), requires_grad=True)
            self.cov_di.data *= np.log(np.e - 1)
            self.softplus = nn.Softplus()
            self.cov_lr = nn.Parameter(torch.ones(num_branches * num_nodes, in_features, rank), requires_grad=True)
            self.cov_lr.data /= math.sqrt(in_features * rank)

    def decide(self, x):
        """
        Compute path probabilities based on the current mean and cov parameters.
        """
        diff = x.unsqueeze(1) - self.weight.unsqueeze(0)
        if self.cov == 'identity':
            return -(diff ** 2).sum(-1) / 2
        elif self.cov == 'diagonal':
            diag_a = self.softplus(self.cov_di)
            return -(diff ** 2 / diag_a + diag_a.log()).sum(-1) / 2
        elif self.cov == 'lowrank':
            diag_i = torch.ones(self.num_nodes * self.num_branches, self.in_features, device=x.device)
            var_inv = woodbury_inverse(diag_i, self.cov_lr, self.rank)
            out1 = diff.unsqueeze(2).matmul(var_inv).matmul(diff.unsqueeze(3)).view(x.size(0), -1)
            out2 = woodbury_logdet(diag_i, self.cov_lr, self.rank)
            return -(out1 + out2) / 2
        elif self.cov == 'both':
            diag_a = self.softplus(self.cov_di)
            var_inv = woodbury_inverse(diag_a, self.cov_lr, self.rank)
            out1 = diff.unsqueeze(2).matmul(var_inv).matmul(diff.unsqueeze(3)).view(x.size(0), -1)
            out2 = woodbury_logdet(diag_a, self.cov_lr, self.rank)
            return -(out1 + out2) / 2
        else:
            raise ValueError(self.kernel)

    def get_covariance(self, index):
        """
        Return the full covariance matrices after reconstruction.
        """
        if self.cov == 'identity':
            return torch.eye(self.in_features, self.in_features)
        elif self.cov == 'diagonal':
            return torch.diag(self.softplus(self.cov_di)[index])
        elif self.cov == 'lowrank':
            mat_u = self.cov_lr[index]
            return torch.eye(self.rank, self.rank) + mat_u.matmul(mat_u.t())
        elif self.cov == 'both':
            mat_a = torch.diag(self.softplus(self.cov_di)[index])
            mat_u = self.cov_lr[index]
            return mat_a + mat_u.matmul(mat_u.t())
        else:
            raise ValueError(self.cov)

    def get_parameters(self, node, child):
        """
        Return the mean and cov parameters for a single node.
        """
        assert node < self.num_nodes
        assert child < self.num_branches
        index = node * self.num_branches + child
        mean = self.weight[index].data.detach()
        cov = self.get_covariance(index)
        return mean, cov.detach()

    def forward(self, x, path):
        """
        Perform forward propagation in the layer.
        """
        num_nodes = self.num_nodes
        num_branches = self.num_branches
        prob_out = self.decide(x)
        prob_out = prob_out.reshape(x.size(0), num_nodes, num_branches)
        prob_out = self.softmax(prob_out)
        path_out = path.unsqueeze(2).repeat((1, 1, num_branches))  # N x D x B
        return (path_out * prob_out).view((path_out.size(0), -1))  # N x BD


class GSDT(nn.Module):
    """
    Class for Gaussian Soft Decision Trees (GSDT).
    """

    def __init__(self, in_features, out_classes, depth=2, num_branches=6, cov='both', rank=2):
        """
        Class initializer.
        """
        super().__init__()

        def to_layer(d):
            num_nodes = count_nodes(d, num_branches)
            return Layer(in_features, num_nodes, num_branches, cov, rank)

        self.cov = cov
        self.depth = depth
        self.num_branches = num_branches
        self.leaf = Leaf(count_nodes(depth, num_branches), out_classes)
        self.layers = nn.ModuleList(to_layer(d) for d in range(depth))

    def path(self, x):
        """
        Get path probabilities.
        """
        path = torch.ones((x.size(0), 1), device=x.device)
        for layer in self.layers:
            path = layer(x, path)
        return path

    def path_list(self, x):
        """
        Get a list of path probabilities (not used in the current codes).
        """
        path = torch.ones((x.size(0), 1), device=x.device)
        paths = []
        for layer in self.layers:
            path = layer(x, path)
            paths.append(path)
        return paths

    def forward(self, x):
        """
        Return the predictions for all examples.
        :param x:
        :return:
        """
        selected_leaves = self.path(x).argmax(dim=1)
        all_logits = self.leaf(x)
        return all_logits[torch.arange(x.size(0)), selected_leaves]


class HingeLoss(nn.Module):
    """
    Class implementing the hinge loss for training GSDT.
    """

    def __init__(self, reduction):
        """
        Class initializer.
        :param reduction:
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, p, y):
        """
        Return the loss value.
        """
        if self.reduction == 'none':
            batch_size = p.size(0)
            num_nodes = p.size(1)
            y = y.contiguous().view(-1)
            p = p.contiguous().view(batch_size * num_nodes, -1)
            p_true = p[torch.arange(p.size(0)), y].unsqueeze(1)
            p_zero = torch.zeros_like(p)
            return torch.sum(torch.max(p - p_true + 1, p_zero), dim=1).view(batch_size, num_nodes)
        elif self.reduction == 'mean':
            p_true = p[torch.arange(p.size(0)), y].unsqueeze(1)
            p_zero = torch.zeros_like(p)
            return torch.sum(torch.max(p - p_true + 1, p_zero), dim=1).mean()
        else:
            raise ValueError(self.reduction)


class TreeLoss(nn.Module):
    """
    Class implementing the overall loss function for training GSDT.
    """

    def __init__(self, loss_type='hinge', lamda=0):
        """
        Class initializer.
        """
        super().__init__()
        if loss_type == 'cross-entropy':
            self.cls_loss = torch.nn.CrossEntropyLoss(reduction='none')
        elif loss_type == 'hinge':
            self.cls_loss = HingeLoss(reduction='none')
            self.kld_loss = torch.nn.KLDivLoss(reduction='none')
            self.log_softmax = torch.nn.LogSoftmax(dim=1)
        else:
            raise ValueError()

        self.lamda = lamda
        self.loss_type = loss_type
        self.reg_loss = nn.KLDivLoss(reduction='batchmean')

    def to_cls_loss(self, path, logit, y):
        """
        Compute the supervised loss for better classification.
        """
        n_nodes = logit.size(1)
        if self.loss_type == 'cross-entropy':
            logit = logit.transpose(1, 2)
        if y.ndimension() == 1:
            y_reshaped = y.unsqueeze(1).expand((-1, n_nodes))
            tq = self.cls_loss(logit, y_reshaped)
        else:
            p_logged = self.log_softmax(logit)
            y_reshaped = y.unsqueeze(2).expand((*y.size(), n_nodes))
            tq = self.kld_loss(p_logged, y_reshaped).sum(dim=1)
        return (path * tq).sum(dim=1).mean()

    @staticmethod
    def to_balance_loss(path):
        """
        Compute the path regularizer shown as Equation (10) in the paper.
        """
        mean_path = torch.mean(path, dim=0)
        return (mean_path * torch.log(mean_path)).sum()

    def forward(self, model, x, y):
        """
        Return the loss value.
        """
        path = model.path(x)
        logit = model.leaf(x)
        loss1 = self.to_cls_loss(path, logit, y)
        loss2 = torch.zeros(1, device=x.device)
        if self.lamda > 0:
            loss2 = self.lamda * self.to_balance_loss(path)
        return loss1, loss2
