#!/usr/bin/env python
import numpy as np
import torch
from torch import nn
from copy import deepcopy


def gen_task():
    """
    Random Linear Regression
    """
    beta = np.random.normal(0, 1, 2)
    x = np.random.uniform(0, 1, (10, 1))
    def f(x):
        x = np.reshape(x, (len(x), 1))
        X = np.hstack([np.ones_like(x), x])
        return np.dot(X, beta)

    return f


def train_on_batch(model, x, y, eta_inner=1e-3):
    x = torch.Tensor(x)
    y = torch.Tensor(y)

    model.zero_grad()
    y_hat = model(x)
    loss = torch.mean((y - y_hat) ** 2)
    loss.backward()

    for param in model.parameters():
        param.data -= eta_inner * param.grad.data

    return model


def train_iter(model, n_train=10, n_inner=10, eta_outer=1e-4):
    phi0 = deepcopy(model.state_dict())

    # generate task
    f = gen_task()
    x = np.random.uniform(0, 1, 50)
    y = f(x)

    # sgd on task
    inds = np.random.permutation(50)
    for _ in range(n_inner):
        for start in range(0, len(x), n_train):
            cur_inds = inds[start:start + n_train]
            train_on_batch(model, x[cur_inds], y[cur_inds])

    task_phi = model.state_dict()
    model.load_state_dict({
        name: phi0[name] + (task_phi[name] - phi0[name]) * eta_outer
        for name in phi0
    })

    return model


def train(model):
    for _ in range(1000):
        model = train_iter(model)
        print("training...")

    return model


model = nn.Sequential(
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

train(model)
