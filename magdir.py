import torch
from math import sqrt
from torch.optim import Optimizer
from utils import convex_f
from recursive import Recursive


class MirrorDescent(object):

    def __init__(self, x1, proj=False):
        """
        Mirror Descent on the unit ball
        """
        self.x = x1
        self.lr = 1
        self.grad_sum = 0

    def step(self, grad):
        self.grad_sum += torch.norm(grad)
        self.lr = 1 / sqrt(self.grad_sum) if self.grad_sum > 0 else 0
        self.x.add_(-grad, alpha=self.lr)
        # x_norm = torch.linalg.norm(self.x)
        x_norm = torch.norm(self.x)
        if x_norm > 1:
            self.x = self.x / x_norm

    def get_output(self):
        return self.x


class Magdir(Optimizer):

    def __init__(self, params, eps=1, f_magnitude=Recursive, f_direction=MirrorDescent):
        defaults = dict(eps=eps, f_magnitude=f_magnitude, f_direction=f_direction)
        super(Magdir, self).__init__(params, defaults)

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
                    eps = group['eps']
                    f_magnitude = group['f_magnitude']
                    f_direction = group['f_direction']

                    # initialize learners
                    state['magnitude'] = f_magnitude([torch.Tensor([eps])])
                    state['direction'] = f_direction(p.data)

                magnitude_learner = state['magnitude']
                direction_learner = state['direction']
                st = torch.dot(grad, direction_learner.get_output())
                magnitude_learner.step(grad=st)
                direction_learner.step(grad)

                # update
                z = magnitude_learner.get_output()[0]
                x = direction_learner.get_output()
                p.data = z * x

        return loss


if __name__ == '__main__':
    fun = lambda x: torch.sum(torch.square(x - 5 * torch.ones_like(x)))
    x = torch.zeros(2).requires_grad_(True)
    opt = Magdir([x])
    for i in range(100):
        opt.zero_grad()
        # f = convex_f(x)
        f = fun(x)
        f.backward(create_graph=True, retain_graph=True)
        torch.nn.utils.clip_grad_norm_(x, 1.0)
        opt.step()
        print(x.detach().numpy())
