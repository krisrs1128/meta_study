#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import glob

npys = glob.glob("../*.npy")

patches = [np.load(x).squeeze().transpose() for x in npys]

plt.imshow(patches[0][:, :, :3])
plt.imshow(patches[1][:, :, :3])
plt.imshow(patches[2][:, :, :3])
plt.imshow(patches[3][:, :, :3])
