import torch
from math import sqrt
from torch.optim import Optimizer
from utils import convex_f


class MirrorDescent(object):

    def __init__(self, x1, diam=1, proj=False):
        self.x = x1
        self.diam = diam
        self.proj = proj
        self.lr = 1
        self.grad_sum = 0

    def update(self, grad):
        self.grad_sum += torch.norm(grad)
        self.lr = 1 / sqrt(self.grad_sum)
        self.x.add_(-grad, alpha=self.lr)
        if self.proj:
            print("Not implemented yet!")

    def get_output(self):
        return self.x


class KT1d(object):

    def __init__(self, eps=1):
        self.grad_sum = 0
        self.t = 1
        self.x = 1
        self.eps = eps
        self.beta = 0
        self.max_grad = 1

    # def update(self, grad):
    #     # if abs(grad) > self.max_grad:
    #     #     self.max_grad = abs(grad)
    #     # grad /= self.max_grad
    #     self.grad_sum += grad
    #     self.beta += self.x * grad
    #     self.beta = max(self.beta, 0)
    #     self.x = -self.grad_sum * (self.eps - self.beta) / self.t
    #     self.t += 1

    def update(self, grad):
        self.grad_sum += torch.norm(grad)
        x = self.x - grad * self.eps / sqrt(self.grad_sum)
        self.x = x if x <= .1 else .1

    def get_output(self):
        return self.x


class Magdir(Optimizer):

    def __init__(self, params, eps=1, f_magnitude=KT1d, f_direction=MirrorDescent):
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
                    state['magnitude'] = f_magnitude(eps)
                    state['direction'] = f_direction(p.data)

                magnitude_learner = state['magnitude']
                direction_learner = state['direction']
                st = torch.dot(grad, direction_learner.get_output())
                # print(st)
                magnitude_learner.update(st)
                direction_learner.update(grad)

                # update
                z = magnitude_learner.get_output()
                x = direction_learner.get_output()
                p.data = z * x
                # print(z)
                # print(x)
                # print()

        return loss


if __name__ == '__main__':
    x = torch.zeros([1, 1]).requires_grad_(True)
    opt = Magdir([x])
    # opt = torch.optim.SGD([x], lr=0.6)
    for i in range(100):
        opt.zero_grad()
        f = convex_f(x)
        f.backward(create_graph=True, retain_graph=True)
        torch.nn.utils.clip_grad_norm_(x, 1.0)
        opt.step()
        print(x.detach().numpy())
