import torch
from torch.optim import Optimizer
from utils import convex_f


class Cocob(Optimizer):

    def __init__(self, params, alpha=10.0, eps=1):
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(alpha=alpha, eps=eps)
        super(Cocob, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            eps = group['eps']
            alpha = group['alpha']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = -p.grad.data  # negative gradient
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['w1'] = p.data
                    state['theta'] = torch.zeros_like(p)
                    state['Gt'] = torch.zeros_like(p)
                    state['Lt'] = eps * torch.ones_like(p)
                    state['reward'] = torch.zeros_like(p)

                # old parameters
                w1 = state['w1']
                theta = state['theta']
                Gt = state['Gt']
                Lt = state['Lt']
                reward = state['reward']

                # updates
                abs_grad = torch.abs(grad)
                Lt = torch.max(Lt, abs_grad)
                theta.add_(grad)
                Gt.add_(abs_grad)
                reward = torch.max(reward + (p.data - w1) * grad, torch.zeros_like(reward))
                p.data = w1 + theta / (Lt * (torch.max(Gt + Lt, alpha * Lt))) * (reward + Lt)

                # state update
                state['theta'] = theta
                state['Gt'] = Gt
                state['Lt'] = Lt
                state['reward'] = reward

                # grad = p.grad.data
                # state = self.state[p]

                # if len(state) == 0:
                #     state['gradients_sum'] = torch.zeros_like(p.data)
                #     state['grad_norm_sum'] = torch.zeros_like(p.data)
                #     state['L'] = eps * torch.ones_like(p.data)
                #     state['tilde_w'] = torch.zeros_like(p.data)
                #     state['reward'] = torch.zeros_like(p.data)

                # theta = state['gradients_sum']
                # G = state['grad_norm_sum']
                # tilde_w = state['tilde_w']
                # L = state['L']
                # reward = state['reward']

                # L = torch.max(L, torch.abs(grad))
                # theta.add_(grad)
                # G.add_(torch.abs(grad))
                # reward = torch.max(reward - grad * tilde_w, torch.zeros_like(reward))
                # new_w = -theta / (L * (torch.max(G + L, alpha * L))) * (reward + L)
                # p.data = p.data - tilde_w + new_w
                # tilde_w_update = new_w

                # state['gradients_sum'] = theta
                # state['grad_norm_sum'] = G
                # state['L'] = L
                # state['tilde_w'] = tilde_w_update
                # state['reward'] = reward

        return loss


if __name__ == "__main__":
    x = torch.zeros([1, 1]).requires_grad_(True)
    opt = Cocob([x])
    # opt = torch.optim.SGD([x], lr=0.6)
    for i in range(100):
        opt.zero_grad()
        f = convex_f(x)
        f.backward(create_graph=True, retain_graph=True)
        torch.nn.utils.clip_grad_norm_(x, 1.0)
        opt.step()
        print(x.detach().numpy())
