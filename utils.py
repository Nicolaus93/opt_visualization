import numpy as np
import torch
import matplotlib.pyplot as plt
import math


def convex_f(tensor):
    x = tensor
    return abs(x - 10)


def rastrigin(tensor, lib=torch):
    # https://en.wikipedia.org/wiki/Test_functions_for_optimization
    x, y = tensor
    A = 10
    f = (
        A * 2
        + (x ** 2 - A * lib.cos(x * math.pi * 2))
        + (y ** 2 - A * lib.cos(y * math.pi * 2))
    )
    return f


def rosenbrock(tensor):
    # https://en.wikipedia.org/wiki/Test_functions_for_optimization
    x, y = tensor
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def plot_rosenbrock(show=False):
    x = np.linspace(-2, 2, 250)
    y = np.linspace(-1, 3, 250)
    minimum = (1.0, 1.0)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock([X, Y])
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.contour(X, Y, Z, 90, cmap='jet')
    ax.set_title('Rosenbrock function')
    ax.plot(*minimum, 'gD')
    if show:
        plt.plot()
        plt.show()
    return fig, ax


def plot_rastrigin(show=False):
    x = np.linspace(-4.5, 4.5, 250)
    y = np.linspace(-4.5, 4.5, 250)
    minimum = (0, 0)
    X, Y = np.meshgrid(x, y)
    Z = rastrigin([X, Y], lib=np)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.contour(X, Y, Z, 20, cmap='jet')
    ax.set_title("Rastrigin function")
    ax.plot(*minimum, 'gD')
    if show:
        plt.plot()
        plt.show()
    return fig, ax


if __name__ == '__main__':
    plot_rosenbrock(show=True)
