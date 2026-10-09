"""Genera configs/<grupo>/<nombre>.yaml: un archivo por experimento.

Cada YAML trae el modelo completo (todos los hiperparámetros explícitos), los de entrenamiento y, como
comentario, el resultado medido en validación (leído de outputs/<nombre>_s*/resumen.json).

Uso:  python src/hacer_configs.py
      python src/entrenar.py --config configs/01_final/final_cd.yaml --seed 42
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONF = ROOT / "configs"
OUT = ROOT / "outputs"

ADAPFORMER = {"name": "adapformer", "T": 336, "L": 48, "N": 12, "D": 256, "r": 32, "d": 64, "k": 4, "E": 2,
              "nhead": 8, "H": 512, "dropout": 0.1, "use_ace": True, "acf_mode": "topk", "anchor_last": True,
              "basin_emb": False, "emb_mlp": False}
LSTM = {"name": "lstm", "N": 12, "L": 48, "hidden": 128, "layers": 2, "dropout": 0.1, "anchor_last": True,
        "basin_emb": False}
TRAIN = {"loss": "log", "lam_mmh": 20.0, "no_aux": False, "ema": 0.0, "epochs": 30, "lr": 0.001,
         "gamma": 0.8, "patience": 5, "batch": 512}
S3 = [42, 43, 44]

# (grupo, nombre, descripción, cambios del modelo, cambios de entrenamiento, semillas, run sin sufijo de semilla)
EXPERIMENTOS = [
    ("01_final", "final_cd", "MODELO FINAL entregado: Adapformer con k = 12 (el ACF usa los 12 canales). "
     "Elegido por MSE de validación en la fase 2; se ensamblan las 3 semillas.", {"acf_mode": "cd"}, {}, S3, False),
    ("02_base", "lstm", "Modelo base: LSTM 2×128 con la misma RevIN, anclaje, pérdida y scheduler.",
     None, {}, S3, False),
    ("02_base", "p1_log", "Adapformer con top-k adaptativo (k = 4): el método completo del paper con la "
     "configuración inicial (k = 4, pérdida log, anclaje, γ = 0.8).", {}, {}, S3, False),
    ("03_ablacion", "abl_ci", "Ablación: k = 1 (channel-independent, sólo el caudal).", {"acf_mode": "ci"}, {}, S3, False),
    ("03_ablacion", "cd_noace", "Ablación sobre el modelo final: k = 12 sin ACE.",
     {"acf_mode": "cd", "use_ace": False}, {}, S3, False),
    ("03_ablacion", "abl_noace", "Ablación sobre k = 4: sin ACE.", {"use_ace": False}, {}, S3, False),
    ("03_ablacion", "abl_noaux", "Ablación sobre k = 4: sin la pérdida auxiliar del SimBlock (Ec. 6).",
     {}, {"no_aux": True}, S3, False),
    ("04_mejoras_descartadas", "p1_hib1", "Pérdida híbrida MSE_log + 1·MSE_mm/h.", {}, {"loss": "hibrida", "lam_mmh": 1.0}, S3, False),
    ("04_mejoras_descartadas", "p1_hib3", "Pérdida híbrida MSE_log + 3·MSE_mm/h.", {}, {"loss": "hibrida", "lam_mmh": 3.0}, S3, False),
    ("04_mejoras_descartadas", "p1_logbas", "Token de cuenca: embedding de basin_id como 13.º token.",
     {"basin_emb": True}, {}, S3, False),
    ("04_mejoras_descartadas", "p1_logema", "Promedio móvil exponencial (EMA 0.999) de los pesos.",
     {}, {"ema": 0.999, "patience": 6}, S3, False),
    ("04_mejoras_descartadas", "cd_embmlp", "k = 12 + embedding con rama no lineal residual (Linear→GELU→Linear).",
     {"acf_mode": "cd", "emb_mlp": True}, {}, S3, False),
    ("04_mejoras_descartadas", "cd_r64", "k = 12 + rango de ACE más alto (r = 64, d = 128).",
     {"acf_mode": "cd", "r": 64, "d": 128}, {}, S3, False),
    ("05_exploracion", "af_log", "Exploración (1 semilla): réplica de la versión inicial, pérdida log.", {}, {}, [42], True),
    ("05_exploracion", "af_hib", "Exploración (1 semilla): pérdida híbrida con λ = 20 (mal calibrado).",
     {}, {"loss": "hibrida", "lam_mmh": 20.0}, [42], True),
    ("05_exploracion", "af_mmh", "Exploración (1 semilla): pérdida sólo en mm/h.", {}, {"loss": "mmh"}, [42], True),
    ("05_exploracion", "af_hib_bas", "Exploración (1 semilla): híbrida λ = 20 + token de cuenca.",
     {"basin_emb": True}, {"loss": "hibrida", "lam_mmh": 20.0}, [42], True),
]


def resultado(nombre: str, semillas, sin_sufijo: bool) -> str:
    runs = [nombre] if sin_sufijo else [f"{nombre}_s{s}" for s in semillas]
    v = [json.loads((OUT / r / "resumen.json").read_text())["val"] for r in runs if (OUT / r / "resumen.json").exists()]
    if not v:
        return "sin corridas todavía"
    mse = np.array([x["MSE"] for x in v]); nse = np.array([x["NSE_median"] for x in v])
    sd = f" ± {mse.std(ddof=1):.5f}" if len(v) > 1 else ""
    return f"val MSE {mse.mean():.5f}{sd} · NSE mediana {nse.mean():.3f} ({len(v)} semilla{'s' if len(v) > 1 else ''})"


def main():
    for grupo, nombre, desc, dm, dt, semillas, sin_sufijo in EXPERIMENTOS:
        modelo = dict(LSTM) if dm is None else {**ADAPFORMER, **dm}
        if modelo.get("acf_mode") == "cd":
            modelo["k"] = modelo["N"]          # k = N: el ACF usa todos los canales (el código lo fija igual)
        elif modelo.get("acf_mode") == "ci":
            modelo["k"] = 1
        train = {**TRAIN, **dt}
        if modelo["name"] == "lstm":
            train["no_aux"] = False
        doc = {"nombre": nombre, "descripcion": desc, "semillas": semillas, "model": modelo, "train": train}
        uso = (f"python src/entrenar.py --config configs/{grupo}/{nombre}.yaml --seed 42 --run {nombre}" if sin_sufijo
               else f"python src/entrenar.py --config configs/{grupo}/{nombre}.yaml --seed 42   # → outputs/{nombre}_s42")
        cab = (f"# {nombre} — {desc}\n# Resultado medido: {resultado(nombre, semillas, sin_sufijo)}\n# Uso: {uso}\n")
        p = CONF / grupo / f"{nombre}.yaml"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(cab + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=110), encoding="utf-8")
        print("escrito", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
