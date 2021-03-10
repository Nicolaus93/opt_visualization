import numpy as np
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math


def convex_f(tensor):
    x = tensor
    return abs(x - 10)


def rastrigin(tensor):
    # https://en.wikipedia.org/wiki/Test_functions_for_optimization
    x, y = tensor
    lib = torch if type(x) == torch.Tensor else np
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


def coherent(tensor):
    x, y = tensor
    lib = torch if type(x) == torch.Tensor else np
    r = lib.sqrt(x**2 + y**2)
    if lib == np:
        theta = np.arctan2(y, x)
    else:
        theta = torch.atan2(y, x)
    return (3 + lib.sin(5 * theta) + lib.cos(3 * theta)) * r**2 * (5 / 3 - r)


def weakly_coherent(tensor):
    x1, x2 = tensor
    return x1**2 * x2**2


def plot_coherent(show=False):
    x = np.linspace(-1.5, 1.5, 250)
    y = np.linspace(-1.5, 1.5, 250)
    minimum = (0, 0)
    X, Y = np.meshgrid(x, y)
    Z = coherent([X, Y])
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.contour(X, Y, Z, 90, cmap='jet')
    ax.set_title(r'Coherent function, $f(x) = 3 + \sin(5\theta) + \cos(3\theta) * r^2(5/3-r)$')
    ax.plot(*minimum, 'gD')
    if show:
        plt.plot()
        plt.show()
    return fig, ax


def plot_weakly_coherent(show=False):
    x = np.linspace(-1.5, 1.5, 250)
    y = np.linspace(-1.5, 1.5, 250)
    minimum = (0, 0)
    X, Y = np.meshgrid(x, y)
    Z = weakly_coherent([X, Y])
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.contour(X, Y, Z, 90, cmap='jet')
    ax.set_title('Coherent function, $f(x) = (x_1^2 x_2^2)$')
    ax.plot(*minimum, 'gD')
    if show:
        plt.plot()
        plt.show()
    return fig, ax


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
    Z = rastrigin([X, Y])
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
    plot_weakly_coherent(show=True)
