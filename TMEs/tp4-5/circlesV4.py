import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from tme5 import CirclesData, MNISTData


def init_model(nx, nh, ny, eta):
    model = torch.nn.Sequential(
        torch.nn.Linear(nx, nh),
        torch.nn.Tanh(),
        torch.nn.Linear(nh, ny),
    )
    loss = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=eta)

    return model, loss, optim


def loss_accuracy(loss, Yhat, Y):

    L = loss(Yhat, torch.argmax(Y.long(), dim=1))
    acc = (
        torch.argmax(
            Yhat,
            dim=1) == torch.argmax(
            Y,
            dim=1)).sum() * 100 / len(Y)

    return L, acc


def main_circles():
    # init
    data = CirclesData()
    data.plot_data()
    np.random.seed(42)
    N = data.Xtrain.shape[0]
    inds = np.arange(0, N)
    np.random.shuffle(inds)
    Xtrain = data.Xtrain[inds]
    Ytrain = data.Ytrain[inds]
    Nbatch = 15
    nx = data.Xtrain.shape[1]
    nh = 10
    ny = data.Ytrain.shape[1]
    eta = 0.03

    # Premiers tests, code à modifier
    model, loss, optim = init_model(nx, nh, ny, eta)

    writer = SummaryWriter()
    L, acc = 0, 0

    # TODO apprentissage
    Nepochs = 200
    for i in range(Nepochs):

        for j in range(0, N, Nbatch):
            Xbatch = Xtrain[j:j + Nbatch]
            Ybatch = Ytrain[j:j + Nbatch]
            Yhat = model(Xbatch)
            L, acc = loss_accuracy(loss, Yhat, Ybatch)
            # Calcule les gradients
            optim.zero_grad()
            L.backward()
            optim.step()

        # Loss and Accuracy on Test
        Yhat_test = model(data.Xtest)
        L_test, acc_test = loss_accuracy(loss, Yhat_test, data.Ytest)

        data.plot_loss(L, L_test, acc, acc_test)

    Ygrid = torch.nn.Softmax(dim=1)(model(data.Xgrid))
    data.plot_data_with_grid(Ygrid.detach())

    # attendre un appui sur une touche pour garder les figures
    input("done")


def main_mnist():
    # init
    data = MNISTData()
    np.random.seed(42)
    N = data.Xtrain.shape[0]
    inds = np.arange(0, N)
    np.random.shuffle(inds)
    Xtrain = data.Xtrain[inds]
    Ytrain = data.Ytrain[inds]
    Xtest = data.Xtest
    Ytest = data.Ytest
    Nbatch = 4096
    nx = data.Xtrain.shape[1]
    nh = 10
    ny = data.Ytrain.shape[1]
    eta = 0.05

    # Premiers tests, code à modifier
    model, loss, optim = init_model(nx, nh, ny, eta)

    writer = SummaryWriter()
    L, acc = 0, 0

    # TODO apprentissage
    Nepochs = 200
    for i in range(Nepochs):

        for j in range(0, N, Nbatch):
            Xbatch = Xtrain[j:j + Nbatch]
            Ybatch = Ytrain[j:j + Nbatch]
            Yhat = model(Xbatch)
            L, acc = loss_accuracy(loss, Yhat, Ybatch)
            # Calcule les gradients
            optim.zero_grad()
            L.backward()
            optim.step()

        # Loss and Accuracy on Test
        Yhat_test = model(Xtest)
        L_test, acc_test = loss_accuracy(loss, Yhat_test, Ytest)
        writer.add_scalar('Loss/train', L, i)
        writer.add_scalar('Accuracy/train', acc, i)
        writer.add_scalar('Loss/test', L_test, i)
        writer.add_scalar('Accuracy/test', acc_test, i)

    # attendre un appui sur une touche pour garder les figures
    input("done")


if __name__ == '__main__':
    main_mnist()
