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
    params['Wh'].requires_grad = True

    params['Wy'] = torch.empty(ny, nh)
    torch.nn.init.normal_(params['Wy'], std=0.3)
    params['Wy'].requires_grad = True

    params['bh'] = torch.empty(nh, 1)
    torch.nn.init.normal_(params['bh'], std=0.3)
    params['bh'].requires_grad = True

    params['by'] = torch.empty(ny, 1)
    torch.nn.init.normal_(params['by'], std=0.3)
    params['by'].requires_grad = True

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


def sgd(params, grads, eta):
    # TODO mettre à jour le contenu de params

    with torch.no_grad():
        params['Wy'] -= eta * params['Wy'].grad
        params['Wy'].grad.zero_()
        params['Wh'] -= eta * params['Wh'].grad
        params['Wh'].grad.zero_()
        params['by'] -= eta * params['by'].grad
        params['by'].grad.zero_()
        params['bh'] -= eta * params['bh'].grad
        params['bh'].grad.zero_()

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

    # TODO apprentissage
    Nepochs = 200
    for i in range(Nepochs):

        for j in range(0, N, Nbatch):
            Xbatch = Xtrain[j:j + Nbatch]
            Ybatch = Ytrain[j:j + Nbatch]
            Yhat, outs = forward(params, Xbatch)
            L, acc = loss_accuracy(Yhat, Ybatch)
            # Calcule les gradients
            L.backward()
            params = sgd(params, grads, eta)

        # Loss and Accuracy on Test
        Yhat_test, outs_test = forward(params, data.Xtest)
        L_test, acc_test = loss_accuracy(Yhat_test, data.Ytest)

        data.plot_loss(L, L_test, acc, acc_test)

    Ygrid, outs_grid = forward(params, data.Xgrid)
    data.plot_data_with_grid(Ygrid.detach())

    # attendre un appui sur une touche pour garder les figures
    input("done")
