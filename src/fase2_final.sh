#!/usr/bin/env bash
# Fase 2: modelo base LSTM + ablación de componentes. Aquí apareció el modelo final (k = 12, final_cd):
# mejor MSE de validación en las 3 semillas. (Las corridas final_cd_s* se llamaron abl_cd_s* al entrenarse.)
set -e; cd "$(dirname "$0")/.."; source src/_comun.sh
for s in 42 43 44; do
  run_cfg configs/02_base/lstm.yaml $s
  run_cfg configs/03_ablacion/abl_ci.yaml $s
  run_cfg configs/01_final/final_cd.yaml $s
  run_cfg configs/03_ablacion/abl_noace.yaml $s
  run_cfg configs/03_ablacion/abl_noaux.yaml $s
done
