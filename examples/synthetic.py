import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from dual_branch_gnn import estimate_effects

rng = np.random.default_rng(0)
n, k = 4000, 6
xy = rng.uniform(size=(n, 2))
_, nearest = cKDTree(xy).query(xy, k + 1)
W = sparse.csr_matrix((np.ones(n * k), (np.repeat(np.arange(n), k), nearest[:, 1:].ravel())), shape=(n, n))

X = rng.normal(size=(n, 6))
WX = W @ X / k
T = rng.binomial(1, 1 / (1 + np.exp(-(0.5 * X[:, 0] + 0.5 * WX[:, 1]))))
share = W @ T / k
S = (share > np.median(share)).astype(int)

Y = 1.0 * T + 0.5 * S + X[:, 0] + 0.5 * WX[:, 1] + rng.normal(scale=0.5, size=n)
res = estimate_effects(X, W, T, S, Y)
print(f"continuous outcome: direct {res['direct']:.3f} (true 1.000), spillover {res['spillover']:.3f} (true 0.500)")

expected = rng.gamma(4.0, 15.0, size=n)
baseline = np.exp(0.15 * X[:, 0] - 0.10 * WX[:, 1])
deaths = rng.poisson(expected * baseline * (1 - 0.12 * T - 0.08 * S))
res = estimate_effects(X, W, T, S, deaths, family="poisson", offset=expected)
print(f"count outcome: direct {res['direct']:.3f} (true {-0.12 * baseline.mean():.3f}), "
      f"spillover {res['spillover']:.3f} (true {-0.08 * baseline.mean():.3f}), "
      f"relative risks {res['rr_direct']:.3f} and {res['rr_spillover']:.3f}")
