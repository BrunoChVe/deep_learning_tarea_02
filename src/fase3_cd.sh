#!/usr/bin/env bash
# Fase 3: ablación de ACE sobre el modelo final (k = 12), 3 semillas.
set -e; cd "$(dirname "$0")/.."; source src/_comun.sh
for s in 42 43 44; do run_cfg configs/03_ablacion/cd_noace.yaml $s; done
