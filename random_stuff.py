import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.optim import Optimizer

plt.style.use('seaborn-white')


class Cocob(Optimizer):

    def __init__(self, params, alpha=10.0, weight_decay=0, eps=1e-8):
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(alpha=alpha, eps=eps, weight_decay=weight_decay)
        self._alpha = alpha
        self._eps = eps
        super(Cocob, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # sum of the gradients
                    state['gradients_sum'] = torch.zeros_like(p)
                    # sum of the absolute values of the subgradients
                    state['grad_norm_sum'] = torch.zeros_like(p)
                    # maximum observed scale
                    state['L'] = self._eps * torch.ones_like(p)
                    state['tilde_w'] = torch.zeros_like(p)
                    state['reward'] = torch.zeros_like(p)

                gradients_sum = state['gradients_sum']
                grad_norm_sum = state['grad_norm_sum']
                L = state['L']
                tilde_w = state['tilde_w']
                reward = state['reward']

                if group['weight_decay'] != 0:
                    grad.add_(p.data, alpha=group['weight_decay'])

                # absolute value of current gradient vector
                abs_grad = torch.abs(grad)

                # update parameters
                reward = torch.max(reward - grad * tilde_w, torch.zeros_like(reward))
                den = L * torch.max(grad_norm_sum + L, self._alpha * L)
                x = (gradients_sum / den) * (reward + L)
                p.data.add_(-tilde_w - x)
                tilde_w = -x

                # state update
                state['gradients_sum'].add_(grad)
                state['grad_norm_sum'].add_(abs_grad)
                state['L'] = torch.max(L, abs_grad)
                state['tilde_w'] = tilde_w
                state['reward'] = reward

        return loss


def rosenbrock(tensor):
    # https://en.wikipedia.org/wiki/Test_functions_for_optimization
    x, y = tensor
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def plot_rosenbrok(holder):
    x = np.linspace(-2, 2, 250)
    y = np.linspace(-1, 3, 250)
    minimum = (1.0, 1.0)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock([X, Y])
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(1, 1, 1)
    ax.contour(X, Y, Z, 90, cmap='jet')
    ax.set_title('Rosenbrock function')
    ax.plot(*minimum, 'gD')
    holder.pyplot(fig)


def plot_optimizer(holder, grad_iter):
    x = np.linspace(-2, 2, 250)
    y = np.linspace(-1, 3, 250)
    minimum = (1.0, 1.0)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock([X, Y])
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(1, 1, 1)
    ax.contour(X, Y, Z, 90, cmap='jet')
    ax.set_title('Rosenbrock function')
    ax.plot(*minimum, 'gD')
    iter_x, iter_y = grad_iter[0, :], grad_iter[1, :]
    ax.plot(iter_x, iter_y, color='r', marker='x')
    ax.plot(iter_x[-1], iter_y[-1], 'rD')
    holder.pyplot(fig)


def execute_steps(func, initial_state, optimizer_class, optimizer_config, num_iter=500):
    x = torch.Tensor(initial_state).requires_grad_(True)
    if optimizer_class == Cocob:
        optimizer = optimizer_class([x], eps=optimizer_config['lr'])
    else:
        optimizer = optimizer_class([x], **optimizer_config)
    steps = np.zeros((2, num_iter + 1))
    steps[:, 0] = np.array(initial_state)
    for i in range(1, num_iter + 1):
        optimizer.zero_grad()
        f = func(x)
        f.backward(create_graph=True, retain_graph=True)
        torch.nn.utils.clip_grad_norm_(x, 1.0)
        optimizer.step()
        steps[:, i] = x.detach().numpy()
    return steps


st.title("Convergence of optimizers")
"""
Here we plot some optimizers in Pytorch.
"""

# chart_data = pd.DataFrame(
#     np.random.randn(20, 3),
#     columns=['a', 'b', 'c'])

# st.line_chart(chart_data)

# if st.checkbox('Show dataframe'):
#     chart_data = pd.DataFrame(
#         np.random.randn(20, 3),
#         columns=['a', 'b', 'c'])

#     st.line_chart(chart_data)

placeholder = st.empty()
plot_rosenbrok(placeholder)


option = st.sidebar.selectbox(
    'Which optimizer would you like to test?',
    ['SGD', 'Adam', 'Cocob'])

'You selected:', option

if option == 'Cocob':
    func = rosenbrock
    optimizer_class = Cocob
    initial_state = (-2.0, 2.0)
    lr = 1.0
    steps = execute_steps(func, initial_state, optimizer_class, {'lr': lr}, num_iter=500)
    st.empty()
    plot_optimizer(placeholder, steps)


# left_column, right_column = st.beta_columns(2)
# pressed = left_column.button('Press me?')
# if pressed:
#     right_column.write("Woohoo!")

# expander = st.beta_expander("FAQ")
# expander.write("Here you could put in some really, really long explanations...")

# import time

# 'Starting a long computation...'

# # Add a placeholder
# latest_iteration = st.empty()
# bar = st.progress(0)

# for i in range(100):
#     # Update the progress bar with each iteration.
#     latest_iteration.text(f'Iteration {i+1}')
#     bar.progress(i + 1)
#     time.sleep(0.1)

# '...and now we\'re done!'
