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

                x0 = state['x0']
                theta = state['theta']
                S2 = state['S2']
                Q = state['Q']
                t = state['t']

                ell_t = grad / sqrt(t)
                if ell_t.shape == torch.Size([1, 1]):
                    ell_t_squared = torch.squeeze(ell_t) * torch.squeeze(ell_t)
                else:
                    ell_t_squared = torch.dot(ell_t, ell_t)
                theta.add_(-ell_t)
                S2 += ell_t_squared
                Q += ell_t_squared / S2

                theta_norm = norm(theta)
                if theta_norm <= S2:
                    # p.data = x0 + theta / (2 * S2) * torch.exp(theta_norm**2 / (4 * S2) - Q)
                    p.data = x0 + theta / (2 * S2) * (1 + theta_norm**2 / (4 * S2) - Q)
                else:
                    # p.data = x0 + theta / (2 * theta_norm) * torch.exp(theta_norm / 2 - S2 / 4 - Q)
                    p.data = x0 + theta / (2 * theta_norm) * (1 + theta_norm / 2 - S2 / 4 - Q)

                state['theta'] = theta
                state['S2'] = S2
                state['Q'] = Q

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
