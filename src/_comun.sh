# Funciones compartidas por los scripts fase*.sh (se cargan con `source`).
# run_cfg <yaml> <semilla> [nombre_run]: entrena salvo que outputs/<run>/resumen.json ya exista.
PY="${PY:-python}"
run_cfg () {
  local cfg="$1" seed="$2" run="$3"
  local nombre; nombre=$(basename "$cfg" .yaml)
  run="${run:-${nombre}_s${seed}}"
  [ -f "outputs/$run/resumen.json" ] && { echo "[omitido] $run ya existe"; return; }
  "$PY" src/entrenar.py --config "$cfg" --seed "$seed" --run "$run"
}
