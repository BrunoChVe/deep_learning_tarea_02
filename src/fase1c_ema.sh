#!/usr/bin/env bash
# Fase 1c: la validación oscila mucho entre épocas → ¿ayuda promediar pesos (EMA)? 3 semillas.
set -e; cd "$(dirname "$0")/.."; source src/_comun.sh
for s in 42 43 44; do run_cfg configs/04_mejoras_descartadas/p1_logema.yaml $s; done
