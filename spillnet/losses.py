import torch
import torch.nn.functional as F


def sliced_wasserstein(p, q, n_slices=50):
    if p.shape[0] == 0 or q.shape[0] == 0:
        return p.new_zeros(())
    theta = torch.randn(p.shape[1], n_slices, device=p.device, dtype=p.dtype)
    theta = theta / (theta.norm(dim=0, keepdim=True) + 1e-8)
    p_sorted, _ = torch.sort((p @ theta).t(), dim=1)
    q_sorted, _ = torch.sort((q @ theta).t(), dim=1)
    n = max(p_sorted.shape[1], q_sorted.shape[1])

    def stretch(x):
        if x.shape[1] == n:
            return x
        return F.interpolate(x.unsqueeze(0), size=n, mode="linear", align_corners=True).squeeze(0)

    return (stretch(p_sorted) - stretch(q_sorted)).abs().mean()


def balance(h, group):
    return sliced_wasserstein(h[group == 0], h[group == 1])


def poisson_deviance(mu, y):
    # without the where, the gradient of xlogy at y = 0 is 0/0
    ratio = torch.where(y > 0, y / mu.clamp_min(1e-8), torch.ones_like(mu))
    return (2 * (torch.xlogy(y, ratio) - (y - mu))).mean()
