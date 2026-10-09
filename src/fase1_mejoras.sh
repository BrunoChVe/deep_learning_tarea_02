#!/usr/bin/env bash
# Fase 1: exploración con 1 semilla (pérdidas y token de cuenca). ~2 min por corrida en RTX 4060.
set -e; cd "$(dirname "$0")/.."; source src/_comun.sh
for c in af_log af_hib af_mmh af_hib_bas; do run_cfg configs/05_exploracion/$c.yaml 42 $c; done
