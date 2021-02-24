import argparse
import os

import pandas as pd
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

import data
import models


def parse_args():
    """
    Parse command line arguments for the main script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--device', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str, default='../out')
    parser.add_argument('--save', action='store_true')

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--branch', type=int, default=6)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--lamda', type=float, default=0.001)
    parser.add_argument('--rank', type=int, default=2)
    parser.add_argument('--cov', type=str, default='both')
    parser.add_argument('--loss-type', type=str, default='hinge')
    args = parser.parse_args()
    return args


def to_device(gpu):
    """
    Change the GPU index into a Torch device.
    """
    if gpu is not None and torch.cuda.is_available():
        return torch.device('cuda:{}'.format(gpu))
    else:
        return torch.device('cpu')


def to_loader(*arrays, batch_size, shuffle=False):
    """
    Make a data loader from multiple data arrays.
    """
    tensors = [torch.from_numpy(a) for a in arrays if a is not None]
    return DataLoader(TensorDataset(*tensors), batch_size, shuffle)


def evaluate(model, loader, device):
    """
    Evaluate a prediction model with a given data loader.
    """
    model.eval()
    acc_sum, count = 0, 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        acc_sum += (y_pred.argmax(1) == y).float().mean().item() * x.size(0)
        count += x.size(0)
    return acc_sum / count


def fit_leaves(model, loader, device, optimizer, updates=10):
    """
    Performs the post-optimization of GSDT (Algorithm 1 in the paper).
    """
    model.eval()
    data_list, path_list = [], []
    for x, _ in loader:
        data_list.append(x)
        path_list.append(model.path(x.to(device)).cpu().detach())
    data_all = torch.cat(data_list)
    arr_prob = torch.cat(path_list)
    arrived = torch.argmax(arr_prob, dim=1)

    for l in range(arr_prob.size(1)):
        data_curr = data_all[arrived == l]
        if (arrived == l).sum() > 0:
            model.layers[-1].weight[l].data.copy_(data_curr.mean(dim=0))
        if model.cov != 'identity' and (arrived == l).sum() > 1:
            cov = torch.from_numpy(np.cov(data_curr.cpu().numpy().transpose())).to(device)
            for i in range(updates):
                loss = ((model.layers[-1].get_covariance(l) - cov) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


def main():
    """
    Main script for training and evaluating a GSDT.
    """
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = to_device(args.device)

    trn_x, trn_y, test_x, test_y = data.read_dataset(args.data)
    in_features = trn_x.shape[1]
    out_classes = trn_y.max() + 1
    model = models.GSDT(in_features, out_classes, args.depth, args.branch, args.cov, args.rank).to(device)

    trn_loader = to_loader(trn_x, trn_y, batch_size=args.batch_size, shuffle=True)
    test_loader = to_loader(test_x, test_y, batch_size=args.batch_size)
    loss_func = models.TreeLoss(args.loss_type, lamda=args.lamda)
    optimizer1 = optim.Adam(model.parameters(), lr=args.lr)
    optimizer2 = optim.Adam(model.parameters(), lr=1e-2)

    logs = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss1_sum, loss2_sum, count = 0, 0, 0
        for x, y in trn_loader:
            x = x.to(device)
            y = y.to(device)
            loss1, loss2 = loss_func(model, x, y)

            optimizer1.zero_grad()
            (loss1 + loss2).backward()
            optimizer1.step()

            loss1_sum += loss1.item() * x.size(0)
            loss2_sum += loss2.item() * x.size(0)
            count += x.size(0)

        if epoch == args.epochs // 2:
            fit_leaves(model, trn_loader, device, optimizer2)

        trn_acc = evaluate(model, trn_loader, device)
        test_acc = evaluate(model, test_loader, device)
        logs.append((epoch, loss1_sum / count, loss2_sum / count, trn_acc, test_acc))

    if args.save:
        model_path = '{}/{}/{}.model'.format(args.out, args.data, args.seed)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)

    df = pd.DataFrame(logs)
    log_path = '{}/{}/{}.log'.format(args.out, args.data, args.seed)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    df.to_csv(log_path, index=False, sep='\t', header=False, float_format='%.4f')

    trn_acc = evaluate(model, trn_loader, device)
    test_acc = evaluate(model, test_loader, device)
    result = np.array([trn_acc, test_acc])
    np.save('{}/{}/{}'.format(args.out, args.data, args.seed), result)


if __name__ == '__main__':
    main()
