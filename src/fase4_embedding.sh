#!/usr/bin/env bash
# Fase 4: ¿se puede hacer que Adapformer supere al LSTM? Sobre el modelo final (k = 12), 3 semillas:
#   cd_embmlp : embedding con rama no lineal residual · cd_r64 : rango de ACE más alto (r = 64, d = 128)
set -e; cd "$(dirname "$0")/.."; source src/_comun.sh
for s in 42 43 44; do
  run_cfg configs/04_mejoras_descartadas/cd_embmlp.yaml $s
  run_cfg configs/04_mejoras_descartadas/cd_r64.yaml $s
done
