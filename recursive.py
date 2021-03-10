import torch
from torch.optim import Optimizer
from utils import convex_f


class DiagonalBetting(object):

    def __init__(self, w, eps=1):
        """
        Implements diagonal betting algo.
        """
        self._eps = eps
        self._wealth = eps * torch.ones_like(w)
        self._v = .5 * torch.ones_like(w)
        self._A = 5 * torch.ones_like(w)
        self._z_sum = torch.zeros_like(w)
        self._x = self._v * self._wealth
        self._w = torch.clamp(self._x, -.5, .5)

    def step(self, grad):

        # retrieve params
        wealth = self._wealth
        v = self._v
        A = self._A
        z_sum = self._z_sum
        w_t = self._w
        x_t = self._x

        # define \tilde{g}
        g = grad / 2  # we assume g € [-2, 2]
        try:
            g[g * (x_t - w_t) < 0] = 0
        except IndexError:
            g = 0 if g * (x_t - w_t) < 0 else g

        # update
        wealth.add_(-x_t * g)
        z_t = g / (1 - g * v)
        z_sum.add_(z_t)
        A.add_(z_t**2)
        v = torch.clamp(-2 * z_sum / A, -.5, .5)
        x_t = v * wealth
        w_t = torch.clamp(x_t, -.5, .5)

        # store
        self._wealth = wealth
        self._v = v
        self._A = A
        self._z_sum = z_sum
        self._w = w_t
        self._x = x_t
        return w_t

    def get_output(self):
        return self._w


class SCBetting(object):

    def __init__(self, w, eps=1):
        self._eps = eps
        self._wealth = eps * torch.ones_like(w)
        self._v = .5 * torch.ones_like(w)
        self._sigma = torch.zeros_like(w)
        self._x = self._v * self._wealth
        self._w = torch.clamp(self._x, -.5, .5)

    def step(self, grad):

        # retrieve params
        wealth = self._wealth
        v = self._v
        sigma = self._sigma
        w_t = self._w
        x_t = self._x

        # define \tilde{g}
        g = grad / 2  # we assume g € [-2, 2]
        try:
            g[g * (x_t - w_t) < 0] = 0
        except IndexError:
            g = 0 if g * (x_t - w_t) < 0 else g

        # update
        wealth.add_(-x_t * g)
        z_t = g / (1 - g * v)
        sigma.add_(z_t**2 / (1 + torch.sign(z_t) * z_t / 2)**2)
        sigma[sigma == 0] = float("Inf")
        v = torch.clamp(v - z_t / sigma, -.5, .5)
        x_t = v * wealth
        w_t = torch.clamp(x_t, -.5, .5)

        # store
        self._wealth = wealth
        self._v = v
        self._sigma = sigma
        self._w = w_t
        self._x = x_t
        return w_t

    def get_output(self):
        return self._w


class Recursive(Optimizer):

    def __init__(self, params, eps=1, inner=DiagonalBetting):
        defaults = dict(eps=eps, inner=inner)
        super(Recursive, self).__init__(params, defaults)

    def step(self, grad=None, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            eps = group['eps']
            inner = group['inner']

            for p in group['params']:
                # if p.grad is None:
                #     continue

                grad = p.grad if grad is None else grad
                state = self.state[p]
                if len(state) == 0:
                    init = (torch.ones_like(p) * eps).requires_grad_(True)
                    state["inner_algo"] = inner(init)
                    state["wealth"] = torch.ones_like(p) * eps

                inner_algo = state["inner_algo"]
                wealth = state["wealth"]

                try:
                    wealth.add_(-torch.dot(grad, p.data))
                    v_t = inner_algo.get_output()
                    z_t = grad / (1 - torch.dot(grad, v_t))
                    v_t = inner_algo.step(z_t)
                except RuntimeError:
                    # 1d case
                    wealth.add_(-torch.squeeze(grad) * torch.squeeze(p.data))
                    v_t = inner_algo.get_output()
                    z_t = grad / (1 - torch.squeeze(grad) * torch.squeeze(v_t))
                    v_t = inner_algo.step(z_t)

                p.data = wealth * v_t
                state["wealth"] = wealth

        return loss

    def get_output(self):
        out = []
        for group in self.param_groups:
            out += [p.data for p in group['params']]
        return out


if __name__ == '__main__':
    f = convex_f
    x = torch.zeros([1, 1]).requires_grad_(True)
    opt = Recursive([x])
    for i in range(50):
        opt.zero_grad()
        f = convex_f(x)
        f.backward(create_graph=True, retain_graph=True)
        torch.nn.utils.clip_grad_norm_(x, 1.0)
        opt.step()
        print(x.detach().numpy())
