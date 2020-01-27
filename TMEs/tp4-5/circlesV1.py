import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from tme5 import CirclesData


def init_params(nx, nh, ny):
    params = {}

    # TODO remplir avec les paramètres Wh, Wy, bh, by
    # params["Wh"] = ...
    params['Wh'] = torch.empty(nh, nx)

    torch.nn.init.normal_(params['Wh'], std=0.3)

    params['Wy'] = torch.empty(ny, nh)
    torch.nn.init.normal_(params['Wy'], std=0.3)

    params['bh'] = torch.empty(nh, 1)
    torch.nn.init.normal_(params['bh'], std=0.3)

    params['by'] = torch.empty(ny, 1)
    torch.nn.init.normal_(params['by'], std=0.3)

    return params


def forward(params, X):
    outputs = {}

    # TODO remplir avec les paramètres X, htilde, h, ytilde, yhat
    # outputs["X"] = ...

    outputs['X'] = X
    outputs['htilde'] = outputs['X'].mm(
        params['Wh'].t()) + params['bh'].t().repeat(X.shape[0], 1)
    outputs['h'] = torch.tanh(outputs['htilde'])
    outputs['ytilde'] = outputs['h'].mm(
        params['Wy'].t()) + params['by'].t().repeat(X.shape[0], 1)
    outputs['yhat'] = F.softmax(outputs['ytilde'], dim=1)

    return outputs['yhat'], outputs


def loss_accuracy(Yhat, Y):
    L = 0
    acc = 0

    # TODO
    # _, indsY = torch.max(Y, 1)
    L = - (Y *
           torch.log(Yhat)).sum(axis=1).mean()

    acc = (
        torch.argmax(
            Yhat,
            dim=1) == torch.argmax(
            Y,
            dim=1)).sum() * 100 / len(Y)

    return L, acc


def backward(params, outputs, Y):
    grads = {}

    # TODO remplir avec les paramètres Wy, Wh, by, bh
    # grads["Wy"] = ...

    grad_y_pred = outputs['yhat'] - Y
    grads['Wy'] = grad_y_pred.t().mm(outputs['h'])

    grads['by'] = grad_y_pred.sum(axis=0).unsqueeze(0).t()
    grad_htilde = (grad_y_pred.mm(
        params['Wy']) * (1 - (outputs['h']**2)))
    grads['Wh'] = grad_htilde.t().mm(outputs['X'])
    grads['bh'] = grad_htilde.sum(axis=0).unsqueeze(0).t()

    return grads


def sgd(params, grads, eta):
    # TODO mettre à jour le contenu de params

    params['Wy'] -= eta * grads['Wy']
    params['Wh'] -= eta * grads['Wh']
    params['by'] -= eta * grads['by']
    params['bh'] -= eta * grads['bh']

    return params


if __name__ == '__main__':

    # init
    data = CirclesData()
    data.plot_data()
    np.random.seed(42)

    N = data.Xtrain.shape[0]
    inds = np.arange(0, N)
    np.random.shuffle(inds)
    Xtrain = data.Xtrain[inds]
    Ytrain = data.Ytrain[inds]
    Nbatch = 10
    nx = data.Xtrain.shape[1]
    nh = 10
    ny = data.Ytrain.shape[1]
    eta = 0.03

    # Premiers tests, code à modifier
    params = init_params(nx, nh, ny)

    writer = SummaryWriter()
    L, acc = 0, 0
    grads = dict()

    # TODO apprentissage
    Nepochs = 200
    for i in range(Nepochs):

        for j in range(0, N, Nbatch):
            Xbatch = Xtrain[j:j + Nbatch]
            Ybatch = Ytrain[j:j + Nbatch]
            Yhat, outs = forward(params, Xbatch)
            L, acc = loss_accuracy(Yhat, Ybatch)
            grads = backward(params, outs, Ybatch)
            params = sgd(params, grads, eta)

        # Loss and Accuracy on Test
        Yhat_test, outs_test = forward(params, data.Xtest)
        L_test, acc_test = loss_accuracy(Yhat_test, data.Ytest)

        data.plot_loss(L, L_test, acc, acc_test)
        # writer.add_scalar('Loss/train', L, i)
        # writer.add_scalar('Accuracy/train', acc, i)
        # writer.add_scalar('Loss/test', L_test, i)
        # writer.add_scalar('Accuracy/test', acc_test, i)

    Ygrid, outs_grid = forward(params, data.Xgrid)
    data.plot_data_with_grid(Ygrid)

    # attendre un appui sur une touche pour garder les figures
    input("done")
