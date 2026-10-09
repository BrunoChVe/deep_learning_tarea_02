"""Datos: caché preprocesada en float16 + carga directa a GPU.

¿Por qué una caché?
  - train.h5 pesa ~5 GB y está guardado con chunks de 1 muestra: leerlo cuesta ~80 s cada vez.
  - La GPU de la laptop (RTX 4060, 8 GB) y la RAM (16 GB) no admiten X en float32 (4.1 GB) con holgura.
  - Tras log1p + z-score los valores quedan en un rango pequeño, así que float16 basta (error ~1e-3 relativo).

Se construye UNA vez (`python src/datos.py`) y produce en `cache/`:
  X_{split}.npy      [n, 336, 12] float16  entradas transformadas (log1p en canales 8 y 11, z-score de train)
  y_{split}.npy      [n, 48]      float32  caudal futuro en mm/h (sin transformar)  — train/validation
  yaux_{split}.npy   [n, 48, 11]  float16  meteorología futura transformada (sólo supervisión) — train/validation
  basin_{split}.npy  [n]          int64    id de cuenca
  lastq_{split}.npy  [n]          float32  último caudal observado en mm/h (para la persistencia)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]          # Entrega_Lab2/
ARCHIVOS = ("train.h5", "test.h5", "metadata.json")    # obligatorios; test_targets.csv y leer_datos.py son opcionales


def _buscar_dataset() -> Path:
    """Carpeta con los datos del laboratorio. Se busca, en este orden:
    1) la variable de entorno LAB2_DATA_DIR, 2) Entrega_Lab2/dataset/, 3) dataset/ al lado de Entrega_Lab2/."""
    if os.environ.get("LAB2_DATA_DIR"):        # si se indica explícitamente, se usa sólo esa carpeta
        return Path(os.environ["LAB2_DATA_DIR"])
    candidatos = [ROOT / "dataset", ROOT.parent / "dataset"]
    for c in candidatos:
        if (c / "train.h5").exists():
            return c
    return candidatos[0]


DATASET = _buscar_dataset()


def verificar_dataset() -> None:
    """Falla con un mensaje claro si no están los archivos del laboratorio."""
    faltan = [a for a in ARCHIVOS if not (DATASET / a).exists()]
    if faltan:
        raise FileNotFoundError(
            f"No se encontraron {', '.join(faltan)} en {DATASET}.\n"
            "Coloque la carpeta con train.h5, test.h5 y metadata.json en una de estas ubicaciones:\n"
            f"  1) {ROOT / 'dataset'}\n"
            f"  2) {ROOT.parent / 'dataset'}\n"
            "  3) cualquier carpeta, indicándola con la variable de entorno LAB2_DATA_DIR")

CACHE = ROOT / "cache"
STATS = ROOT / "outputs" / "stats.json"               # media/std por canal calculadas sobre train
TARGET = 11
LOG_CHANNELS = (8, 11)


def cargar_stats():
    with open(STATS, encoding="utf-8") as fh:
        s = json.load(fh)
    return np.array(s["mean"], np.float32), np.array(s["std"], np.float32)


def transformar_X(x: np.ndarray, mean, std) -> np.ndarray:
    """x [.., T, 12] en unidades originales → log1p en canales 8 y 11 → z-score por canal."""
    x = np.nan_to_num(x.astype(np.float32, copy=True))
    for c in LOG_CHANNELS:
        x[..., c] = np.log1p(np.clip(x[..., c], 0, None))
    return (x - mean) / std


def transformar_aux(a: np.ndarray, mean, std) -> np.ndarray:
    """y_aux [.., L, 11] (canales 0–10) → mismo espacio que la entrada."""
    a = np.nan_to_num(a.astype(np.float32, copy=True))
    a[..., 8] = np.log1p(np.clip(a[..., 8], 0, None))
    return (a - mean[:11]) / std[:11]


def construir_cache(block: int = 20_000) -> None:
    """Lee los H5 por bloques contiguos y escribe los .npy (memoria pico < 1 GB)."""
    verificar_dataset()
    print(f"[cache] dataset: {DATASET}", flush=True)
    CACHE.mkdir(exist_ok=True)
    mean, std = cargar_stats()
    t0 = time.time()
    with h5py.File(DATASET / "train.h5", "r") as f:
        split = f["split"][:]
        basin = f["basin_id"][:].astype(np.int64)
        n_tot = len(split)
        out = {}
        for code, name in ((0, "train"), (1, "validation")):
            n = int((split == code).sum())
            out[name] = {
                "X": np.lib.format.open_memmap(CACHE / f"X_{name}.npy", "w+", np.float16, (n, 336, 12)),
                "y": np.lib.format.open_memmap(CACHE / f"y_{name}.npy", "w+", np.float32, (n, 48)),
                "yaux": np.lib.format.open_memmap(CACHE / f"yaux_{name}.npy", "w+", np.float16, (n, 48, 11)),
                "lastq": np.empty(n, np.float32), "pos": 0}
            np.save(CACHE / f"basin_{name}.npy", basin[split == code])
        for s in range(0, n_tot, block):
            e = min(s + block, n_tot)
            X = f["X"][s:e]; y = f["y"][s:e]; a = f["y_aux"][s:e]; sp = split[s:e]
            Xt = transformar_X(X, mean, std).astype(np.float16)
            At = transformar_aux(a, mean, std).astype(np.float16)
            for code, name in ((0, "train"), (1, "validation")):
                m = sp == code
                k = int(m.sum()); o = out[name]; p = o["pos"]
                o["X"][p:p + k] = Xt[m]; o["y"][p:p + k] = y[m]; o["yaux"][p:p + k] = At[m]
                o["lastq"][p:p + k] = np.clip(X[m, -1, TARGET], 0, None)
                o["pos"] += k
            print(f"  train.h5 {e}/{n_tot}  ({time.time() - t0:.0f}s)", flush=True)
        for name, o in out.items():
            for k in ("X", "y", "yaux"):
                o[k].flush()
            np.save(CACHE / f"lastq_{name}.npy", o["lastq"])
    with h5py.File(DATASET / "test.h5", "r") as f:
        n = f["X"].shape[0]
        Xo = np.lib.format.open_memmap(CACHE / "X_test.npy", "w+", np.float16, (n, 336, 12))
        lq = np.empty(n, np.float32)
        for s in range(0, n, block):
            X = f["X"][s:s + block]
            Xo[s:s + len(X)] = transformar_X(X, mean, std).astype(np.float16)
            lq[s:s + len(X)] = np.clip(X[:, -1, TARGET], 0, None)
        Xo.flush()
        np.save(CACHE / "lastq_test.npy", lq)
        np.save(CACHE / "basin_test.npy", f["basin_id"][:].astype(np.int64))
    # Objetivos reales del test (opcionales; sólo para el reporte final, nunca para elegir modelos)
    if (DATASET / "test_targets.csv").exists():
        import pandas as pd
        yt = pd.read_csv(DATASET / "test_targets.csv").sort_values("Id")
        np.save(CACHE / "y_test.npy", yt.drop(columns="Id").to_numpy(np.float32))
    else:
        print("[cache] aviso: no hay test_targets.csv; se omitirán las métricas de test", flush=True)
    print(f"[cache] lista en {time.time() - t0:.0f}s -> {CACHE}")


def cargar(split: str, device: str = "cuda", aux: bool = False, max_samples: int | None = None,
           seed: int = 0) -> dict:
    """Carga un split de la caché a `device`. X queda en float16 (se convierte a fp32 por batch)."""
    d = {}
    X = np.load(CACHE / f"X_{split}.npy", mmap_mode="r")
    idx = np.arange(X.shape[0])
    if max_samples and max_samples < len(idx):
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_samples, replace=False))
    sub = (lambda a: a[idx]) if max_samples else (lambda a: a)
    # Copia a GPU por bloques (evita duplicar 2 GB en RAM)
    Xg = torch.empty((len(idx), 336, 12), dtype=torch.float16, device=device)
    for s in range(0, len(idx), 20_000):
        Xg[s:s + 20_000] = torch.from_numpy(np.ascontiguousarray(X[idx[s:s + 20_000]])).to(device)
    d["X"] = Xg
    for k in ("y", "basin", "lastq"):
        p = CACHE / f"{k}_{split}.npy"
        if p.exists():
            d[k] = torch.from_numpy(np.ascontiguousarray(sub(np.load(p)))).to(device)
    if aux:
        A = np.load(CACHE / f"yaux_{split}.npy", mmap_mode="r")
        d["yaux"] = torch.from_numpy(np.ascontiguousarray(A[idx] if max_samples else A[:])).to(device)
    d["n"] = len(idx)
    return d


def mapa_cuencas() -> dict:
    """basin_id → índice 0..507 (las 508 cuencas de train también están en validación y test)."""
    ids = np.unique(np.load(CACHE / "basin_train.npy"))
    return {int(b): i for i, b in enumerate(ids)}


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # consolas de Windows sin UTF-8
    construir_cache()
