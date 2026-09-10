"""A tiny neural scenario generator for the 'explore' projections.

`ReturnGenerator` is an autoregressive **mixture-density network** written in
plain NumPy - one hidden layer, a 2-component Gaussian-mixture head. It learns
p(next monthly return | recent returns, recent volatility) and can then roll
that forward to produce simulated return paths.

Honest scope: with only ~15-20 years of monthly data (~200 points) this model
learns a little structure - the average, roughly how volatility clusters, a hint
of fat tails - and not much more. It is offered next to the plain historical
resampling as a teaching comparison, not because it is more trustworthy. It is
not, for data this small.
"""

from __future__ import annotations

import numpy as np


class ReturnGenerator:
    def __init__(self, n_lags: int = 3, vol_window: int = 6, hidden: int = 8,
                 n_mix: int = 2, l2: float = 3e-4, seed: int = 0):
        self.n_lags = n_lags
        self.vol_window = vol_window
        self.hidden = hidden
        self.K = n_mix
        self.l2 = l2
        self.rng = np.random.default_rng(seed)
        self.warmup = max(n_lags, vol_window)
        self._fitted = False

    # ------------------------------------------------------------------ #
    # features: [lag1, lag2, lag3, mean_6m, std_6m]
    # ------------------------------------------------------------------ #
    def _design(self, s: np.ndarray):
        X, y = [], []
        for i in range(self.warmup, len(s)):
            hist = s[:i]
            lags = hist[-self.n_lags:][::-1]
            win = hist[-self.vol_window:]
            X.append(np.concatenate([lags, [win.mean(), win.std() + 1e-6]]))
            y.append(s[i])
        return np.asarray(X, float), np.asarray(y, float)

    def _forward(self, Xn: np.ndarray):
        h = np.tanh(Xn @ self.W1 + self.b1)
        out = h @ self.W2 + self.b2
        K = self.K
        a_pi = out[:, :K]
        mu = out[:, K:2 * K]
        log_sig = np.clip(out[:, 2 * K:], -4.0, 3.0)
        sig = np.exp(log_sig)
        pi = np.exp(a_pi - a_pi.max(1, keepdims=True))
        pi /= pi.sum(1, keepdims=True)
        return h, pi, mu, sig, log_sig

    def fit(self, monthly_pct, epochs: int = 600, lr: float = 0.02):
        s = np.asarray(monthly_pct, float)
        X, y = self._design(s)
        if len(X) < 24:
            self._fitted = False
            return self

        self.x_mean, self.x_std = X.mean(0), X.std(0) + 1e-6
        Xn = (X - self.x_mean) / self.x_std
        F = Xn.shape[1]

        r = self.rng
        self.W1 = r.normal(0, 0.3, (F, self.hidden))
        self.b1 = np.zeros(self.hidden)
        self.W2 = r.normal(0, 0.2, (self.hidden, 3 * self.K))
        self.b2 = np.zeros(3 * self.K)

        params = ["W1", "b1", "W2", "b2"]
        m = {p: np.zeros_like(getattr(self, p)) for p in params}
        v = {p: np.zeros_like(getattr(self, p)) for p in params}
        b1_, b2_, eps = 0.9, 0.999, 1e-8
        yb = y[:, None]
        n = len(y)

        for t in range(1, epochs + 1):
            h, pi, mu, sig, log_sig = self._forward(Xn)
            logpdf = -0.5 * ((yb - mu) / sig) ** 2 - log_sig - 0.5 * np.log(2 * np.pi)
            joint = np.log(pi + 1e-12) + logpdf
            mx = joint.max(1, keepdims=True)
            logmix = mx[:, 0] + np.log(np.exp(joint - mx).sum(1))
            gamma = np.exp(joint - logmix[:, None])

            d_api = (pi - gamma)
            d_mu = gamma * (mu - yb) / sig ** 2
            d_logsig = gamma * (1.0 - ((yb - mu) / sig) ** 2)
            dout = np.concatenate([d_api, d_mu, d_logsig], axis=1) / n

            grads = {
                "W2": h.T @ dout + self.l2 * self.W2,
                "b2": dout.sum(0),
            }
            dh = (dout @ self.W2.T) * (1 - h ** 2)
            grads["W1"] = Xn.T @ dh + self.l2 * self.W1
            grads["b1"] = dh.sum(0)

            for p in params:
                g = grads[p]
                m[p] = b1_ * m[p] + (1 - b1_) * g
                v[p] = b2_ * v[p] + (1 - b2_) * g * g
                mhat = m[p] / (1 - b1_ ** t)
                vhat = v[p] / (1 - b2_ ** t)
                setattr(self, p, getattr(self, p) - lr * mhat / (np.sqrt(vhat) + eps))

        self._fitted = True
        return self

    def sample_paths(self, warmup_pct, n_months: int, n_paths: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        s = np.asarray(warmup_pct, float)
        lo, hi = s.min() * 1.8 - 1.0, s.max() * 1.8 + 1.0
        tail = np.tile(s[-self.warmup:], (n_paths, 1))
        out = np.empty((n_paths, n_months))

        for t in range(n_months):
            lags = tail[:, -self.n_lags:][:, ::-1]
            win = tail[:, -self.vol_window:]
            X = np.concatenate(
                [lags, win.mean(1, keepdims=True), win.std(1, keepdims=True) + 1e-6], axis=1
            )
            Xn = (X - self.x_mean) / self.x_std
            _, pi, mu, sig, _ = self._forward(Xn)
            comp = (rng.random((n_paths, 1)) < np.cumsum(pi, 1)).argmax(1)
            rows = np.arange(n_paths)
            draw = np.clip(rng.normal(mu[rows, comp], sig[rows, comp]), lo, hi)
            out[:, t] = draw
            tail = np.concatenate([tail, draw[:, None]], axis=1)
        return out


def generate_paths(monthly_returns, n_months: int, n_paths: int, seed: int = 0):
    """Fit the generator on decimal monthly returns and sample return paths.

    Returns ``(paths, faithfulness)`` where ``paths`` is an
    ``(n_paths, n_months)`` array of decimal monthly returns, or
    ``(None, {"ok": False})`` if there isn't enough history to train.
    """
    m = np.asarray(monthly_returns, float)
    m = m[~np.isnan(m)]
    if len(m) < 40:
        return None, {"ok": False}

    pct = m * 100.0
    gen = ReturnGenerator(seed=seed).fit(pct)
    if not gen._fitted:
        return None, {"ok": False}

    paths = gen.sample_paths(pct, n_months, n_paths, seed=seed) / 100.0
    # The model has no business having a view on the average return - a small
    # autoregressive net trained on ~200 points drifts. Anchor the overall mean
    # to history and let the model keep only the *shape* (clustering, tails).
    paths = paths - paths.mean() + float(m.mean())
    faith = {
        "ok": True,
        "n_train": int(len(m)),
        "hist_mean": float(m.mean()), "gen_mean": float(paths.mean()),
        "hist_std": float(m.std()), "gen_std": float(paths.std()),
        "hist_min": float(m.min()), "gen_min": float(paths.min()),
    }
    return paths, faith
