import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from tme5 import CirclesData


def init_params(nx, nh, ny):
    params = {}

    # TODO remplir avec les paramètres Wh, Wy, bh, by
    # params["Wh"] = ...
    params['Wh'] = torch.empty(nx, nh)

    torch.nn.init.normal_(params['Wh'], std=0.3)

    params['Wy'] = torch.empty(nh, ny)
    torch.nn.init.normal_(params['Wy'], std=0.3)

    params['bh'] = torch.empty(1, nh)
    torch.nn.init.normal_(params['bh'], std=0.3)

    params['by'] = torch.empty(1, ny)
    torch.nn.init.normal_(params['by'], std=0.3)

    return params


def forward(params, X):
    outputs = {}

    # TODO remplir avec les paramètres X, htilde, h, ytilde, yhat
    # outputs["X"] = ...

    outputs['X'] = X
    outputs['htilde'] = outputs['X'].mm(params['Wh']) + params['bh']
    outputs['h'] = F.tanh(outputs['htilde'])
    outputs['ytilde'] = outputs['h'].mm(params['Wy'])
    outputs['yhat'] = F.softmax(outputs['ytilde'])

    return outputs['yhat'], outputs


def loss_accuracy(Yhat, Y):
    L = 0
    acc = 0

    # TODO
    _, indsY = torch.max(Y, 1)
    L = - torch.mean(Y[indsY] * torch.log(Yhat[indsY]))
    acc = (Yhat == Y).sum() * 100 / len(Y)

    return L, acc


def backward(params, outputs, Y):
    grads = {}

    # TODO remplir avec les paramètres Wy, Wh, by, bh
    # grads["Wy"] = ...

    grad_y_pred = None
    grad_h_softmax = None
    grads['Wy'] = None
    grads['Wh'] = None
    grads['by'] = torch.ones(1, params['Wy'].shape[0])
    grads['bh'] = torch.ones(1, params['Wh'].shape[0])

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
    N = data.Xtrain.shape[0]
    Nbatch = 10
    nx = data.Xtrain.shape[1]
    nh = 10
    ny = data.Ytrain.shape[1]
    eta = 0.03

    # Premiers tests, code à modifier
    params = init_params(nx, nh, ny)
    Yhat, outs = forward(params, data.Xtrain)

    L, _ = loss_accuracy(Yhat, data.Ytrain)

    grads = backward(params, outs, data.Ytrain)
    params = sgd(params, grads, eta)

    # TODO apprentissage

    # attendre un appui sur une touche pour garder les figures
    input("done")
