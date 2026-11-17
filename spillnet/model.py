import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformFirstBlock(nn.Module):
    """Returns [B1, B2] with B1 = ReLU(W ReLU(Z theta0) theta1) and B2 = W B1 theta2."""

    def __init__(self, d_in, hidden, d_out):
        super().__init__()
        self.theta0, self.theta1, self.theta2 = (nn.Linear(d_in, hidden), nn.Linear(hidden, hidden),
                                                 nn.Linear(hidden, d_out))

    def forward(self, Z, W):
        b1 = F.relu(self.theta1(torch.sparse.mm(W, F.relu(self.theta0(Z)))))
        return torch.cat([b1, self.theta2(torch.sparse.mm(W, b1))], 1)


class Branch(nn.Module):
    """MLP on the input concatenated with its transform-first graph block."""

    def __init__(self, d_in, hidden, d_block, mlp_hidden, d_out):
        super().__init__()
        self.block = TransformFirstBlock(d_in, hidden, d_block)
        self.mlp = nn.Sequential(nn.Linear(d_in + hidden + d_block, mlp_hidden), nn.ReLU(),
                                 nn.Linear(mlp_hidden, d_out))

    def forward(self, Z, W):
        return self.mlp(torch.cat([Z, self.block(Z, W)], 1))


class OutcomeHeads(nn.Module):
    """One head per combination of own treatment and neighbourhood exposure, columns (T, S) = 00, 01, 10, 11."""

    def __init__(self, d_in, n_heads=4, hidden1=64, hidden2=32, dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(d_in, hidden1), nn.ReLU(), nn.Dropout(dropout),
                          nn.Linear(hidden1, hidden2), nn.ReLU(), nn.Dropout(dropout),
                          nn.Linear(hidden2, 1))
            for _ in range(n_heads)])

    def forward(self, h):
        return torch.stack([head(h).squeeze(-1) for head in self.heads], dim=1)


class SpillNet(nn.Module):
    """Local branch on X, neighbourhood branch on WX, four outcome heads and two propensity heads."""

    def __init__(self, n_covariates, family="gaussian"):
        super().__init__()
        self.family = family
        self.local = Branch(n_covariates, 32, 32, 64, 32)
        self.neighbour = Branch(n_covariates, 32, 16, 32, 16)
        self.heads = OutcomeHeads(32 + 16)
        self.propensity_T = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
        self.propensity_S = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1))

    def encode(self, X, WX, W):
        h_local, h_neighbour = self.local(X, W), self.neighbour(WX, W)
        return torch.cat([h_local, h_neighbour], 1), h_local, h_neighbour

    def potentials(self, h):
        out = self.heads(h)
        return F.softplus(out) if self.family == "poisson" else out
