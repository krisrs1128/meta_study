#!/usr/bin/env python
"""
Autoencoder for Tasks

The idea is to use latent representation of the tiles to define tasks.
"""
import torch
import numpy as np
from torch.utils.data import DataLoader


def train_epoch(model, iterator, optimizer, loss_fun):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epoch_loss = 0
    model.train()

    for x, _, _ in iterator:
        optimizer.zero_grad()
        x = x.to(device)

        _, _, x_hat = model(x)
        loss = loss_fun(x_hat, x)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return model, epoch_loss / len(iterator)

model = ConvAutoencoder()
optimizer = torch.optim.Adam(model.parameters())
loss_fun = nn.MSELoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
loss_fun = loss_fun.to(device)

train_iter = DataLoader(LCMPatches("/data/"))

for epoch in range(10):
    print("beginning epoch {}".format(epoch))
    model, train_loss = train_epoch(model, train_iter, optimizer, loss_fun)
