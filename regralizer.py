import torch
from math import sqrt
from torch.optim import Optimizer
from utils import convex_f
from torch import norm


class Regralizer(Optimizer):

    def __init__(self, params):
        """
        Implements FTRL with rescaled gradients and linearithmic regularizer.
        """
        defaults = {}
        super(Regralizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            # we should check gradients are bounded

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['x0'] = p.data
                    state['theta'] = torch.zeros_like(grad)
                    state['S2'] = 4
                    state['Q'] = 0
                    state['t'] = 1

                # retrieve params
                x0 = state['x0']
                theta = state['theta']
                S2 = state['S2']
                Q = state['Q']
                t = state['t']

                # update
                ell_t = grad / sqrt(t)
                ell_t_squared = norm(ell_t)**2
                theta.add_(-ell_t)
                S2 += ell_t_squared
                Q += ell_t_squared / S2

                theta_norm = norm(theta)
                if theta_norm <= S2:
                    p.data = x0 + theta / (2 * S2) * torch.exp(theta_norm**2 / (4 * S2) - Q)
                else:
                    p.data = x0 + theta / (2 * theta_norm) * torch.exp(theta_norm / 2 - S2 / 4 - Q)

                # store params
                state['theta'] = theta
                state['S2'] = S2
                state['Q'] = Q
                state['t'] = t + 1

        return loss


if __name__ == '__main__':
    f = convex_f
    x = torch.zeros([1, 1]).requires_grad_(True)
    opt = Regralizer([x])
    for i in range(100):
        opt.zero_grad()
        f = convex_f(x)
        f.backward(create_graph=True, retain_graph=True)
        torch.nn.utils.clip_grad_norm_(x, 1.0)
        opt.step()
        print(x.detach().numpy())
