import numpy as np
import torch
from scipy import sparse

from .training import fit

SEED_STEP = 100003


def row_normalise(W):
    W = sparse.csr_matrix(W, dtype=np.float64)
    degree = np.asarray(W.sum(1)).ravel()
    return sparse.diags(np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)) @ W


def split(n, seed):
    order = np.random.default_rng(seed).permutation(n)
    n_train, n_val = int(n * 0.70), int(n * 0.15)
    return order[:n_train], order[n_train:n_train + n_val]


def to_tensors(X, W, device="cpu"):
    W = row_normalise(W)
    WX = np.asarray(W @ X)
    coo = W.tocoo()
    W_t = torch.sparse_coo_tensor(torch.from_numpy(np.vstack([coo.row, coo.col])).long(),
                                  torch.from_numpy(coo.data).float(), size=W.shape).coalesce()
    dense = lambda M: torch.from_numpy(np.ascontiguousarray(M)).float().to(device)
    return dense(X), dense(WX), W_t.to(device)


def estimate_effects(X, W, T, S, Y, family="gaussian", offset=None, mask=None, seed=777, n_seeds=3, device="cpu"):
    """Direct and spillover effects of a binary treatment on a network.

    X is an (n, p) covariate array and W an (n, n) adjacency or weights matrix, row-normalised here. T is own
    treatment and S neighbourhood exposure, both 0 or 1, and Y is the outcome. With family="poisson", Y holds
    counts and offset the expected counts or person-time. mask marks the units with outcome data, and the other
    units stay in the network. Returns the average direct and spillover effects over the masked units, per-unit
    direct_i and spillover_i, the (n, 4) potentials with columns (T, S) = 00, 01, 10, 11 and, for counts, the
    relative risks rr_direct and rr_spillover. All are averaged over n_seeds training seeds.
    """
    if family not in ("gaussian", "poisson"):
        raise ValueError("family must be 'gaussian' or 'poisson'")
    if family == "poisson" and offset is None:
        raise ValueError("family='poisson' needs an offset")
    X = np.asarray(X)
    T, S, Y = np.asarray(T).astype(int), np.asarray(S).astype(int), np.asarray(Y, dtype=np.float64)
    offset = None if offset is None else np.asarray(offset, dtype=np.float64)
    units = np.arange(len(T)) if mask is None else np.flatnonzero(mask)
    train, val = (units[i] for i in split(len(units), seed))
    X_t, WX_t, W_t = to_tensors(X, W, device)
    P = np.mean([fit(X_t, WX_t, W_t, T, S, Y, train, val, seed + SEED_STEP * s, family, offset, device)
                 for s in range(n_seeds)], axis=0, dtype=np.float64)
    i = np.arange(len(T))
    direct_i, spillover_i = P[i, 2 + S] - P[i, S], P[i, 2 * T + 1] - P[i, 2 * T]
    out = dict(direct=float(direct_i[units].mean()), spillover=float(spillover_i[units].mean()),
               direct_i=direct_i, spillover_i=spillover_i, potentials=P)
    if family == "poisson":
        out.update(rr_direct=float(P[i, 2 + S][units].mean() / P[i, S][units].mean()),
                   rr_spillover=float(P[i, 2 * T + 1][units].mean() / P[i, 2 * T][units].mean()))
    return out
