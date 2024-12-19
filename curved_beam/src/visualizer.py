import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


def viz_2D_mesh(Xp):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), dpi=200)
    axes.set_aspect('equal', adjustable='box')
    plt.plot(Xp[:, 0], Xp[:, 1], 'o')
    plt.savefig('../out/sampling_points.png')
    plt.show()


def plot_loss(result):
    plt.figure(figsize=(4, 4), dpi=200)
    plt.semilogy(range(len(result)), result)
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.savefig('../out/loss.png')
    plt.show()

