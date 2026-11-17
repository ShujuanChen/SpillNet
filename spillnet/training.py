import numpy as np
import torch
import torch.nn.functional as F

from .losses import balance, poisson_deviance
from .model import SpillNet

ALPHA = 0.01
BETA = 1.0
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 1e-5
MAX_GRAD_NORM = 5.0
PATIENCE = 50
MAX_EPOCHS = 2000


def fit(X, WX, W, T, S, Y, train, val, seed, family="gaussian", offset=None, device="cpu"):
    """Trains one model and returns its (n, 4) potential outcomes."""
    torch.manual_seed(seed)
    np.random.seed(seed % 2**31)
    model = SpillNet(X.shape[1], family).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    T_t = torch.from_numpy(T.astype(np.int64)).to(device)
    S_t = torch.from_numpy(S.astype(np.int64)).to(device)
    y = torch.from_numpy(Y.astype(np.float32)).to(device)
    off = torch.from_numpy(offset.astype(np.float32)).to(device) if family == "poisson" else None
    tr, va = torch.from_numpy(train).to(device), torch.from_numpy(val).to(device)
    cell = (2 * T_t + S_t).unsqueeze(1)

    def factual(h):
        return model.potentials(h).gather(1, cell).squeeze(1)

    def outcome_loss(pred, rows):
        if family == "poisson":
            return poisson_deviance(off[rows] * pred[rows], y[rows])
        return ((pred - y)[rows] ** 2).mean()

    def propensity_loss(h_local, h_neighbour):
        return (F.binary_cross_entropy_with_logits(model.propensity_T(h_local).squeeze(1)[tr], T_t.float()[tr])
                + F.binary_cross_entropy_with_logits(model.propensity_S(h_neighbour).squeeze(1)[tr], S_t.float()[tr]))

    best, best_state, wait = float("inf"), None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        opt.zero_grad()
        h, h_local, h_neighbour = model.encode(X, WX, W)
        loss = outcome_loss(factual(h), tr)
        loss = loss + ALPHA * (balance(h_local, T_t) + balance(h_neighbour, S_t))
        loss = loss + BETA * propensity_loss(h_local, h_neighbour)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(outcome_loss(factual(model.encode(X, WX, W)[0]), va))
        if val_loss < best - 1e-6:
            best, wait = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model.potentials(model.encode(X, WX, W)[0]).cpu().numpy()
