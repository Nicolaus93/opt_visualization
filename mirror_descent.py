import torch
from math import sqrt
from torch.optim import Optimizer
from utils import convex_f


class MirrorDescent(Optimizer):

    def __init__(self, params, diam=1):
        """
        Implements Mirror Descent without projection.
        """
        defaults = dict(diam=diam)
        super(MirrorDescent, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            diam = group['diam']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['grad_sum'] = torch.norm(grad)
                else:
                    state['grad_sum'].add_(torch.norm(grad))

                grad_sum = state['grad_sum']
                p.data.add_(-grad, alpha=diam / sqrt(grad_sum))
                # p_norm = torch.norm(p.data)
                # if p_norm > 10e6:
                #     p.data = p.data * 10e6 / p_norm

        return loss


if __name__ == '__main__':
    f = convex_f
    x = torch.zeros([1, 1]).requires_grad_(True)
    opt = MirrorDescent([x])
    # opt = torch.optim.SGD([x], lr=0.6)
    for i in range(100):
        opt.zero_grad()
        f = convex_f(x)
        f.backward(create_graph=True, retain_graph=True)
        torch.nn.utils.clip_grad_norm_(x, 1.0)
        opt.step()
        print(x.detach().numpy())
