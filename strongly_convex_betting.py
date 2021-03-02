import torch
from math import sqrt
from torch.optim import Optimizer
from utils import convex_f
from torch import norm


class SCBetting(Optimizer):

    def __init__(self, params, eps=1):
        """
        Implements Strongly-convex (MD) betting algo.
        """
        defaults = dict(eps=eps)
        super(SCBetting, self).__init__(params, defaults)

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

        return loss


if __name__ == '__main__':
    f = convex_f
    x = torch.zeros([1, 1]).requires_grad_(True)
    opt = SCBetting([x])
    for i in range(100):
        opt.zero_grad()
        f = convex_f(x)
        f.backward(create_graph=True, retain_graph=True)
        torch.nn.utils.clip_grad_norm_(x, 1.0)
        opt.step()
        print(x.detach().numpy())
