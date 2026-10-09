"""Entrenamiento (Adapformer o LSTM) sobre la caché fp16, pensado para una GPU de 8 GB.

Uso (desde Entrega_Lab2/). La forma recomendada es con un YAML de configs/:
  python src/entrenar.py --config configs/01_final/final_cd.yaml --seed 42      # → outputs/final_cd_s42
También se puede todo por línea de comandos:
  python src/entrenar.py --run af_base   --model adapformer --loss log
  python src/entrenar.py --run af_hib    --model adapformer --loss hibrida --basin-emb
  python src/entrenar.py --run lstm_hib  --model lstm       --loss hibrida --basin-emb
  python src/entrenar.py --run af_k1 --model adapformer --loss hibrida --basin-emb --set acf_mode=ci

Pérdidas (`--loss`):
  log     : MSE en el espacio log1p + z-score (la pérdida del modelo final). Trata igual un error del 10 %
            en un caudal de 0.01 mm/h que en uno de 5 mm/h → subestima picos.
  mmh     : MSE directamente en mm/h (la MISMA métrica con la que se evalúa).
  hibrida : MSE_log + λ·MSE_mm/h. El término log cuida las cuencas de caudal bajo (NSE por cuenca),
            el término en mm/h cuida las crecidas (que concentran ~98 % del MSE).
Ec. 6 del paper: + λ_aux·‖W_y − W_dec‖²/N si el modelo es Adapformer y `--no-aux` no se usa.

Salidas en outputs/<run>/: best.pt, log.csv, pred_val.npy, pred_test.npy, resumen.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

V2 = Path(__file__).resolve().parent
sys.path.insert(0, str(V2))
from datos import CACHE, ROOT, cargar, cargar_stats, mapa_cuencas  # noqa: E402
from modelos import TARGET, construir  # noqa: E402
from metricas import compute_metrics  # noqa: E402  (MSE, MAE, NSE/KGE por cuenca…)

OUT = ROOT / "outputs"
MEAN, STD = cargar_stats()
M11, S11 = float(MEAN[TARGET]), float(STD[TARGET])
Z_MAX = (math.log1p(50.0) - M11) / S11          # tope numérico (50 mm/h, el máximo observado es 24)


def a_mmh(z):
    """Espacio del modelo (log1p + z-score) → caudal en mm/h."""
    return torch.expm1(z.clamp(max=Z_MAX) * S11 + M11).clamp(min=0)


def a_z(q):
    """Caudal en mm/h → espacio del modelo."""
    return (torch.log1p(q.clamp(min=0)) - M11) / S11


def perdida(out, y, yaux, cfg, revin=None):
    z_hat = out["y"]
    parts = {}
    if cfg["loss"] in ("log", "hibrida"):
        parts["log"] = F.mse_loss(z_hat, a_z(y))
    if cfg["loss"] in ("mmh", "hibrida"):
        parts["mmh"] = F.mse_loss(a_mmh(z_hat), y)
    total = parts.get("log", 0) + cfg["lam_mmh"] * parts.get("mmh", 0) if cfg["loss"] == "hibrida" \
        else parts.get("log", parts.get("mmh"))
    if cfg["aux"] and out["W_dec"] is not None and yaux is not None:
        # W_y: similitud entre canales del FUTURO real (y_aux + y), normalizado con la RevIN del histórico
        fut = torch.cat([yaux.float(), a_z(y)[..., None]], -1)
        fut = revin.normalize_future(fut)
        w_y = torch.softmax(torch.einsum("bln,blm->bnm", fut, fut) / fut.shape[1], -1)
        n = w_y.shape[-1]
        parts["aux"] = ((w_y - out["W_dec"]) ** 2).sum((1, 2)).mean() / n
        total = total + parts["aux"]
    return total, parts


@torch.no_grad()
def predecir(model, d, bidx, bs=4096):
    """Predicción por lotes → mm/h [n, 48] en numpy."""
    model.eval()
    out = []
    for s in range(0, d["n"], bs):
        xb = d["X"][s:s + bs].float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model(xb, bidx[s:s + bs])["y"]
        out.append(a_mmh(z.float()).cpu())
    return torch.cat(out).numpy()


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # consolas de Windows sin UTF-8
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="YAML de configs/ (modelo + entrenamiento); la semilla y --set se dan por línea de comandos")
    ap.add_argument("--run", default=None, help="nombre de la corrida (por defecto <nombre del YAML>_s<semilla>)")
    ap.add_argument("--model", default="adapformer", choices=["adapformer", "lstm"])
    ap.add_argument("--loss", default="hibrida", choices=["log", "mmh", "hibrida"])
    ap.add_argument("--lam-mmh", type=float, default=20.0)
    ap.add_argument("--basin-emb", action="store_true")
    ap.add_argument("--no-aux", action="store_true", help="quita la pérdida auxiliar del SimBlock (Ec. 6)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ema", type=float, default=0.0,
                    help="decaimiento del promedio móvil exponencial de los pesos (0 = desactivado, p. ej. 0.999)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--max-samples", type=int, default=None, help="submuestreo para pruebas rápidas")
    ap.add_argument("--set", nargs="*", default=[], help="hiperparámetros del modelo, p. ej. k=8 acf_mode=ci")
    a = ap.parse_args()

    mcfg = {"name": a.model, "basin_emb": a.basin_emb}
    if a.config:
        # El YAML define el modelo completo y los hiperparámetros de entrenamiento (sección `train`,
        # mismas claves que las opciones de línea de comandos con "_" en vez de "-").
        import yaml
        with open(a.config, encoding="utf-8") as fh:
            y = yaml.safe_load(fh)
        # Lo que se pasa explícitamente por línea de comandos tiene prioridad sobre el YAML
        explicitos = {act.dest for act in ap._actions if any(o in sys.argv for o in act.option_strings)}
        for k, v in y.get("train", {}).items():
            if not hasattr(a, k):
                raise ValueError(f"clave de train desconocida en {a.config}: {k}")
            if k not in explicitos:
                setattr(a, k, v)
        mcfg = dict(y["model"])
        a.model = mcfg["name"]
        if a.run is None:
            a.run = f"{y['nombre']}_s{a.seed}"
    if a.run is None:
        ap.error("falta --run (o --config)")

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda"
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        mcfg[k] = v
    cfg = {"model": mcfg, "loss": a.loss, "lam_mmh": a.lam_mmh, "aux": a.model == "adapformer" and not a.no_aux,
           "seed": a.seed, "lr": a.lr, "gamma": a.gamma, "patience": a.patience, "batch": a.batch,
           "epochs": a.epochs, "max_samples": a.max_samples, "ema": a.ema}
    rdir = OUT / a.run
    rdir.mkdir(parents=True, exist_ok=True)
    print(f"[{a.run}] cfg={json.dumps(cfg)}", flush=True)

    t0 = time.time()
    tr = cargar("train", dev, aux=cfg["aux"], max_samples=a.max_samples, seed=a.seed)
    va = cargar("validation", dev)
    te = cargar("test", dev)
    cmap = mapa_cuencas()
    lut = torch.full((max(cmap) + 1,), 0, dtype=torch.long, device=dev)
    for b, i in cmap.items():
        lut[b] = i
    bi = {k: lut[d["basin"]] for k, d in (("tr", tr), ("va", va), ("te", te))}
    print(f"[{a.run}] datos en GPU ({time.time() - t0:.0f}s): train={tr['n']} val={va['n']} test={te['n']}", flush=True)

    model = construir(mcfg).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, a.gamma)
    # EMA: copia del modelo cuyos pesos son un promedio móvil de los pesos entrenados. Suaviza la
    # oscilación entre batches/épocas; se usa para validar, guardar y predecir.
    ema = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(a.ema)) if a.ema > 0 else None
    evalm = ema.module if ema is not None else model
    g = torch.Generator(device=dev).manual_seed(a.seed)
    y_va, b_va = va["y"].cpu().numpy(), va["basin"].cpu().numpy()

    best, best_ep, bad = float("inf"), 0, 0
    with open(rdir / "log.csv", "w", newline="") as fh:
        csv.writer(fh).writerow(["epoca", "loss", "log", "mmh", "aux", "val_mse", "val_mae", "val_nse_med", "lr", "seg"])
    for ep in range(1, a.epochs + 1):
        te0 = time.time()
        model.train()
        acc = {"loss": 0.0, "log": 0.0, "mmh": 0.0, "aux": 0.0}
        perm = torch.randperm(tr["n"], device=dev, generator=g)
        nb = 0
        for s in range(0, tr["n"], a.batch):
            idx = perm[s:s + a.batch]
            xb = tr["X"][idx].float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(xb, bi["tr"][idx])
            loss, parts = perdida(out, tr["y"][idx], tr["yaux"][idx] if cfg["aux"] else None, cfg, model.revin)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if ema is not None:
                ema.update_parameters(model)
            acc["loss"] += loss.item() if nb % 50 == 0 else 0  # muestreo barato del log
            for k, v in parts.items():
                acc[k] += v.item() if nb % 50 == 0 else 0
            nb += 1
        lr = opt.param_groups[0]["lr"]
        sch.step()
        p_va = predecir(evalm, va, bi["va"])
        m = compute_metrics(y_va, p_va, b_va, full=False)
        k50 = max(1, (nb + 49) // 50)
        row = [ep] + [round(acc[k] / k50, 5) for k in ("loss", "log", "mmh", "aux")] + \
              [m["MSE"], m["MAE"], m["NSE_median"], lr, round(time.time() - te0, 1)]
        with open(rdir / "log.csv", "a", newline="") as fh:
            csv.writer(fh).writerow(row)
        mejora = m["MSE"] < best
        print(f"[{a.run}] ep {ep:2d} loss={row[1]:.4f} | val MSE={m['MSE']:.6f} MAE={m['MAE']:.5f} "
              f"NSEmed={m['NSE_median']:.4f} lr={lr:.1e} {row[-1]:.0f}s{' *' if mejora else ''}", flush=True)
        if mejora:
            best, best_ep, bad = m["MSE"], ep, 0
            torch.save({"model": evalm.state_dict(), "cfg": cfg, "epoch": ep}, rdir / "best.pt")
        else:
            bad += 1
            if bad >= a.patience:
                break

    evalm.load_state_dict(torch.load(rdir / "best.pt", weights_only=False)["model"])
    p_va = predecir(evalm, va, bi["va"])
    p_te = predecir(evalm, te, bi["te"])
    np.save(rdir / "pred_val.npy", p_va.astype(np.float32))
    np.save(rdir / "pred_test.npy", p_te.astype(np.float32))
    mv = compute_metrics(y_va, p_va, b_va, full=True)
    mt = None
    if (CACHE / "y_test.npy").exists():          # test_targets.csv es opcional
        y_te = np.load(CACHE / "y_test.npy")
        mt = compute_metrics(y_te, p_te, te["basin"].cpu().numpy(), full=True)
    res = {"run": a.run, "cfg": cfg, "n_params": n_par, "best_epoch": best_ep, "epochs_run": ep,
           "minutos": round((time.time() - t0) / 60, 2), "val": mv, "test": mt}
    with open(rdir / "resumen.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"[{a.run}] FIN mejor época {best_ep} | VAL MSE={mv['MSE']:.6f} NSEmed={mv['NSE_median']:.4f} "
          + (f"| TEST MSE={mt['MSE']:.6f} NSEmed={mt['NSE_median']:.4f} " if mt else "")
          + f"| {res['minutos']} min", flush=True)


if __name__ == "__main__":
    main()
