"""Agregación de resultados: tablas por corrida, media ± σ entre semillas, ensambles y pruebas por cuenca."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from datos import CACHE, ROOT
from metricas import compute_metrics, per_basin_nse_kge

OUT = ROOT / "outputs"
COLS = ["MSE", "MAE", "NSE_median", "NSE_p25", "KGE_median", "MSE_h1", "MSE_h24", "MSE_h48",
        "MSE_flood", "MSE_normal", "peak_rel_err_flood"]


def verdad(split: str):
    """(y [n,48] mm/h, basin [n], último caudal [n]) del split 'validation' o 'test'."""
    return (np.load(CACHE / f"y_{split}.npy"), np.load(CACHE / f"basin_{split}.npy"),
            np.load(CACHE / f"lastq_{split}.npy"))


def corridas(patron: str = ".*") -> list[str]:
    return sorted(p.parent.name for p in OUT.glob("*/resumen.json") if re.fullmatch(patron, p.parent.name))


def resumen(run: str) -> dict:
    return json.loads((OUT / run / "resumen.json").read_text())


def tabla(runs: list[str], split: str = "val") -> pd.DataFrame:
    filas = {}
    for r in runs:
        s = resumen(r)
        filas[r] = {**{c: s[split].get(c) for c in COLS}, "mejor_época": s["best_epoch"], "min": s["minutos"]}
    return pd.DataFrame(filas).T


def por_config(runs: list[str], split: str = "val") -> pd.DataFrame:
    """Agrupa corridas `<config>_s<semilla>` → media y σ de MSE / NSE mediana / KGE mediana."""
    t = tabla(runs, split)
    t["config"] = [re.sub(r"_s\d+$", "", r) for r in t.index]
    g = t.groupby("config")[["MSE", "NSE_median", "KGE_median"]].agg(["mean", "std", "count"])
    return g


def pred(run: str, split: str = "val") -> np.ndarray:
    return np.load(OUT / run / ("pred_val.npy" if split == "val" else "pred_test.npy"))


def persistencia(split: str = "validation") -> np.ndarray:
    y, _, q = verdad(split)
    return np.repeat(q[:, None], y.shape[1], 1)


def metricas_de(p: np.ndarray, split: str = "validation") -> dict:
    y, b, _ = verdad(split)
    return compute_metrics(y, p, b, full=True)


def ensamble(runs: list[str], split: str = "val") -> np.ndarray:
    """Promedio de las predicciones (en mm/h) de varias corridas."""
    return np.mean([pred(r, split) for r in runs], 0)


def nse_por_cuenca(p: np.ndarray, split: str = "validation") -> pd.Series:
    y, b, _ = verdad(split)
    ids, nse, _ = per_basin_nse_kge(y, p, b)
    return pd.Series(nse, index=ids)


def wilcoxon_cuencas(p_a: np.ndarray, p_b: np.ndarray, split: str = "validation") -> dict:
    """¿A gana a B en NSE por cuenca? Prueba de Wilcoxon pareada sobre las 508 cuencas."""
    from scipy.stats import wilcoxon
    a, b = nse_por_cuenca(p_a, split), nse_por_cuenca(p_b, split)
    m = a.notna() & b.notna()
    d = (a - b)[m]
    return {"gana_A_%": float(100 * (d > 0).mean()), "dif_mediana": float(d.median()),
            "p_valor": float(wilcoxon(a[m], b[m]).pvalue)}
