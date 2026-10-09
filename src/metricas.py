"""Métricas en mm/h: MSE, MAE, RMSE globales; NSE y KGE por cuenca; por horizonte y por régimen.

NSE y KGE por cuenca se calculan juntando todas las ventanas y horizontes de la cuenca
(las ventanas de validación se solapan en el tiempo; es la aproximación disponible sin fechas).
"""
from __future__ import annotations

import numpy as np

HORIZONS = (1, 6, 12, 24, 48)


def _np(a):
    try:
        import torch
        if isinstance(a, torch.Tensor):
            return a.detach().float().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(a, dtype=np.float32)


def per_basin_nse_kge(y: np.ndarray, p: np.ndarray, basin: np.ndarray):
    """NSE y KGE por cuenca. y, p [n, L]; basin [n]. Devuelve (ids, nse, kge)."""
    ids, inv = np.unique(basin, return_inverse=True)
    L = y.shape[1]
    g = np.repeat(inv, L)
    yf, pf = y.ravel().astype(np.float64), p.ravel().astype(np.float64)
    nb = len(ids)
    cnt = np.bincount(g, minlength=nb)
    sy, sp = np.bincount(g, yf, nb), np.bincount(g, pf, nb)
    syy, spp, syp = np.bincount(g, yf * yf, nb), np.bincount(g, pf * pf, nb), np.bincount(g, yf * pf, nb)
    my, mp = sy / cnt, sp / cnt
    vy, vp = syy / cnt - my ** 2, spp / cnt - mp ** 2
    cov = syp / cnt - my * mp
    sse = syy - 2 * syp + spp
    with np.errstate(divide="ignore", invalid="ignore"):
        nse = 1 - sse / (vy * cnt)
        r = cov / np.sqrt(vy * vp)
        alpha = np.sqrt(vp / vy)
        beta = mp / my
        kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    valid = vy > 1e-12
    nse[~valid] = np.nan; kge[~valid] = np.nan
    return ids, nse, kge


def compute_metrics(y, p, basin=None, full: bool = True, flood_threshold: float | None = None) -> dict:
    """Métricas en mm/h. y, p [n, L]. Si `full`, añade por horizonte y por régimen."""
    y, p = _np(y).astype(np.float64), _np(p).astype(np.float64)
    err = p - y
    m = {"MSE": float((err ** 2).mean()), "MAE": float(np.abs(err).mean())}
    m["RMSE"] = float(np.sqrt(m["MSE"]))
    m["NSE_global"] = float(1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum())
    if basin is not None:
        b = basin.detach().cpu().numpy() if hasattr(basin, "detach") else np.asarray(basin)
        _, nse, kge = per_basin_nse_kge(y, p, b)
        m["NSE_median"] = float(np.nanmedian(nse)); m["NSE_p25"] = float(np.nanpercentile(nse, 25))
        m["KGE_median"] = float(np.nanmedian(kge)); m["KGE_p25"] = float(np.nanpercentile(kge, 25))
    if full:
        for h in HORIZONS:
            e = err[:, h - 1]
            m[f"MSE_h{h}"] = float((e ** 2).mean()); m[f"MAE_h{h}"] = float(np.abs(e).mean())
        ymax = y.max(axis=1)
        thr = float(np.quantile(ymax, 0.9)) if flood_threshold is None else flood_threshold
        fl = ymax > thr
        m["flood_threshold"] = thr
        for name, mask in (("flood", fl), ("normal", ~fl)):
            e = err[mask]
            m[f"MSE_{name}"] = float((e ** 2).mean()); m[f"MAE_{name}"] = float(np.abs(e).mean())
            m[f"bias_{name}"] = float(e.mean())
            # error relativo en el pico: (max pred − max real)/max real
            m[f"peak_rel_err_{name}"] = float(np.median((p[mask].max(1) - y[mask].max(1)) /
                                                         np.maximum(y[mask].max(1), 1e-6)))
    return m


def mse_by_horizon(y, p) -> np.ndarray:
    y, p = _np(y), _np(p)
    return ((p - y) ** 2).mean(axis=0)
