"""Modelos: Adapformer (Luo et al., 2026) adaptado a caudal + LSTM base.

Implementación propia en un solo archivo, con una extensión opcional (probada y descartada en la ablación):

  * `basin_emb=True` → **token de cuenca**: un vector aprendido por cuenca (nn.Embedding) que se añade
    como un token extra a la secuencia del encoder.
  * `emb_mlp=True` → rama no lineal residual en el embedding.

Flujo de formas (B = batch, T = 336, N = 12, D = 256, L = 48):

  x [B,T,N] ─RevIN─► x_norm [B,T,N] ─SimBlock─► W_dec [B,N,N]
                         │
                         └─transponer─► [B,N,T] ─Linear(T→D)─► x_emb [B,N,D] ─ACE─► [B,N,D]
                                                     (+ token cuenca → [B,N+1,D])
                         ─Encoder Transformer (atención entre canales)─► x_enc [B,N(+1),D]
                         ─ACF: top-k de W_dec[caudal,:] → gather [B,k,D] → MLP → [B,k,L] → fila 0
                         ─RevIN⁻¹ (anclado al último caudal)─► ŷ [B,L]  (espacio log1p + z-score)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

TARGET = 11  # índice del canal de caudal


# ----------------------------------------------------------------------------------- RevIN
class RevIN(nn.Module):
    """Normalización de instancia reversible (Kim et al., 2021) — Ec. 1 del paper.

    Cada ventana se normaliza con SU media y desviación (por canal) → el modelo ve formas, no niveles.
    Al final se revierte sólo para el canal objetivo. `anchor_last=True` (adaptación nuestra):
    la salida se interpreta como desviación respecto al ÚLTIMO caudal observado, no respecto a la media.
    """

    def __init__(self, n: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n))
        self.bias = nn.Parameter(torch.zeros(n))

    def norm(self, x):
        self.mean = x.mean(1, keepdim=True).detach()
        self.std = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + self.eps).detach()
        self.last = x[:, -1:, :].detach()
        return (x - self.mean) / self.std * self.weight + self.bias

    def denorm_target(self, y, anchor_last: bool):
        c = TARGET
        y = (y - self.bias[c]) / (self.weight[c] + 1e-10)
        ref = self.last if anchor_last else self.mean
        return y * self.std[:, 0, c:c + 1] + ref[:, 0, c:c + 1]

    def normalize_future(self, y_fut):
        """Normaliza el futuro [B,L,N] con las estadísticas del histórico (para W_y de la Ec. 6)."""
        return (y_fut - self.mean) / self.std


# ----------------------------------------------------------------------------------- SimBlock
class SimBlock(nn.Module):
    """Ec. 4–5: W_enc = XᵀX/T (similitud entre canales), W_dec = softmax(W_enc + ReLU(MLP(W_enc)))."""

    def __init__(self, n: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(n, 4 * n), nn.GELU(), nn.Linear(4 * n, n))

    def forward(self, x_norm):
        w_enc = torch.einsum("btn,btm->bnm", x_norm, x_norm) / x_norm.shape[1]
        return torch.softmax(w_enc + F.relu(self.mlp(w_enc)), dim=-1)


# ----------------------------------------------------------------------------------- ACE
class ACE(nn.Module):
    """Adaptive Channel Enhancer (Alg. 1, l. 6–12): realce de bajo rango compartido.

    W ∈ ℝ^{D×r} → W_enh = ReLU(Linear(r→d)(W)) ∈ ℝ^{D×d}; x_low = x_emb·W_enh; salida = Linear([x_emb, x_low]).
    Intuición: proyecta cada token a d "modos temporales dominantes" y los reinyecta en el token.
    """

    def __init__(self, D: int, r: int = 32, d: int = 64):
        super().__init__()
        self.W = nn.Parameter(torch.empty(D, r))
        nn.init.xavier_uniform_(self.W)
        self.mlp = nn.Linear(r, d)
        self.fuse = nn.Linear(D + d, D)

    def forward(self, x_emb):
        w_enh = F.relu(self.mlp(self.W)).to(x_emb.dtype)
        return self.fuse(torch.cat([x_emb, x_emb @ w_enh], dim=-1))


# ----------------------------------------------------------------------------------- ACF
class ACF(nn.Module):
    """Adaptive Channel Forecaster (Ec. 2–3) para el caudal.

    modo "topk": C = [caudal, k−1 canales con mayor W_dec[caudal,:]] (selección por muestra)
    modo "ci"  : sólo el caudal (k = 1, channel-independent)
    modo "cd"  : los 12 canales en orden fijo (channel-dependent)
    El predictor (MLP) recibe los k tokens aplanados y devuelve k filas de L valores; se usa la fila 0.
    """

    def __init__(self, mode: str, k: int, N: int, D: int, H: int, L: int, dropout: float):
        super().__init__()
        self.mode = mode
        self.k = 1 if mode == "ci" else (N if mode == "cd" else k)
        order = [TARGET] + [c for c in range(N) if c != TARGET]
        self.register_buffer("orden", torch.tensor(order), persistent=False)
        self.L = L
        self.net = nn.Sequential(nn.Linear(self.k * D, H), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(H, self.k * L))

    def seleccionar(self, w_dec):
        B = w_dec.shape[0]
        tgt = torch.full((B, 1), TARGET, dtype=torch.long, device=w_dec.device)
        if self.mode == "topk" and self.k > 1:
            s = w_dec[:, TARGET, :].clone()
            s[:, TARGET] = float("-inf")                # el caudal siempre va primero
            return torch.cat([tgt, s.topk(self.k - 1, dim=-1).indices], 1)
        if self.k == 1:
            return tgt
        return self.orden[None].expand(B, -1)

    def forward(self, x_enc, w_dec):
        C = self.seleccionar(w_dec)                                            # [B,k]
        x_c = torch.gather(x_enc, 1, C[..., None].expand(-1, -1, x_enc.shape[-1]))   # [B,k,D]
        y = self.net(x_c.flatten(1)).view(-1, self.k, self.L)
        return y[:, 0], C


# ----------------------------------------------------------------------------------- Adapformer
class Adapformer(nn.Module):
    def __init__(self, T=336, L=48, N=12, D=256, r=32, d=64, k=4, E=2, nhead=8, H=512, dropout=0.1,
                 use_ace=True, acf_mode="topk", anchor_last=True, basin_emb=False, n_basins=508,
                 emb_mlp=False, **_):
        super().__init__()
        self.anchor_last, self.N = anchor_last, N
        self.revin = RevIN(N)
        self.simblock = SimBlock(N)
        self.embed = nn.Linear(T, D)
        # emb_mlp=True (mejora probada en la fase 4): rama no lineal residual sobre el embedding lineal,
        # e = Linear(x); e = e + Linear(GELU(e)). Sigue siendo 1 token por canal, con más capacidad que una
        # sola capa lineal para describir la forma de las 336 h (el LSTM recorre la serie hora a hora).
        self.embed_mlp = nn.Sequential(nn.GELU(), nn.Linear(D, D)) if emb_mlp else None
        self.drop = nn.Dropout(dropout)
        self.ace = ACE(D, r, d) if use_ace else None
        self.basin = nn.Embedding(n_basins, D) if basin_emb else None
        if self.basin is not None:
            nn.init.normal_(self.basin.weight, std=0.02)
        layer = nn.TransformerEncoderLayer(D, nhead, 2 * D, dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, E, norm=nn.LayerNorm(D), enable_nested_tensor=False)
        self.acf = ACF(acf_mode, k, N, D, H, L, dropout)

    def forward(self, x, basin_idx=None):
        with torch.autocast("cuda", enabled=False):
            x_norm = self.revin.norm(x.float())
            w_dec = self.simblock(x_norm)
        h = self.embed(x_norm.transpose(1, 2))                      # [B,N,D] cada canal = 1 token
        if self.embed_mlp is not None:
            h = h + self.embed_mlp(h)
        h = self.drop(h)
        if self.ace is not None:
            h = self.ace(h)
        if self.basin is not None:
            h = torch.cat([h, self.basin(basin_idx)[:, None, :].to(h.dtype)], 1)   # [B,N+1,D]
        h = self.encoder(h)[:, : self.N]                             # descartamos el token de cuenca
        y_norm, C = self.acf(h, w_dec)
        with torch.autocast("cuda", enabled=False):
            y = self.revin.denorm_target(y_norm.float(), self.anchor_last)
        return {"y": y, "W_dec": w_dec, "sel": C}


# ----------------------------------------------------------------------------------- LSTM base
class LSTMBase(nn.Module):
    """Modelo base: LSTM de 2 capas sobre las 12 variables (misma RevIN) + Linear(hidden→48)."""

    def __init__(self, N=12, L=48, hidden=128, layers=2, dropout=0.1, anchor_last=True,
                 basin_emb=False, n_basins=508, **_):
        super().__init__()
        self.anchor_last = anchor_last
        self.revin = RevIN(N)
        self.lstm = nn.LSTM(N, hidden, layers, batch_first=True, dropout=dropout)
        self.basin = nn.Embedding(n_basins, hidden) if basin_emb else None
        self.head = nn.Linear(hidden, L)

    def forward(self, x, basin_idx=None):
        with torch.autocast("cuda", enabled=False):
            x_norm = self.revin.norm(x.float())
        out, _ = self.lstm(x_norm)
        h = out[:, -1]
        if self.basin is not None:
            h = h + self.basin(basin_idx).to(h.dtype)
        y_norm = self.head(h)
        with torch.autocast("cuda", enabled=False):
            y = self.revin.denorm_target(y_norm.float(), self.anchor_last)
        return {"y": y, "W_dec": None, "sel": None}


def construir(cfg: dict) -> nn.Module:
    m = dict(cfg)
    name = m.pop("name")
    return {"adapformer": Adapformer, "lstm": LSTMBase}[name](**m)
