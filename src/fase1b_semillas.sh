#!/usr/bin/env bash
# Fase 1b: las diferencias de la fase 1 son del orden del ruido → 3 semillas por configuración.
# λ de la pérdida híbrida recalibrado: en train el MSE en mm/h es ~0.1 (no ~0.012 como en validación).
set -e; cd "$(dirname "$0")/.."; source src/_comun.sh
for s in 42 43 44; do
  run_cfg configs/02_base/p1_log.yaml $s
  run_cfg configs/04_mejoras_descartadas/p1_hib1.yaml $s
  run_cfg configs/04_mejoras_descartadas/p1_hib3.yaml $s
  run_cfg configs/04_mejoras_descartadas/p1_logbas.yaml $s
done
