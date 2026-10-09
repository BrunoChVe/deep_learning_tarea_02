# Laboratorio 2 — Pronóstico de caudal con Adapformer

Implementación propia en PyTorch de *Adapformer: Adaptive channel management for multivariate time series
forecasting* (Luo et al., Neural Networks 193, 2026), adaptada a pronosticar el caudal específico (mm/h) de las
próximas 48 h a partir de 336 h de 12 variables.

**Empieza por:** `INFORME.md` y `notebooks/00_guia.ipynb`.

## Antes de ejecutar: dónde va el dataset

Los datos del laboratorio no vienen en esta entrega. Coloque la carpeta del dataset en **una** de estas ubicaciones
(se buscan en este orden):

1. La ruta indicada en la variable de entorno `LAB2_DATA_DIR`. Si está definida, se usa solo esa.
2. `Entrega_Lab2/dataset/`, dentro de esta carpeta.
3. `dataset/`, al lado de esta carpeta.

La carpeta debe contener:

| Archivo | ¿Obligatorio? | Para qué |
|---|---|---|
| `train.h5`, `test.h5`, `metadata.json` | Sí | Entrenamiento, validación y predicción de test |
| `test_targets.csv` | No | Métricas de test (reporte final) y verificación del formato |
| `leer_datos.py` | No | Ejemplo de lectura en el notebook 01 |

Si los archivos no están donde se espera, `python src/datos.py` se detiene con un mensaje que muestra las tres
ubicaciones válidas.

## Resultado

| Modelo | Val MSE | Test MSE |
|---|---|---|
| Persistencia | 0.01901 | 0.01411 |
| Adapformer k = 4 (top-k adaptativo), ensamble de 3 semillas | 0.01192 | 0.01109 |
| **Adapformer k = 12, ensamble de 3 semillas (entregado)** | **0.01117** | **0.01086** |
| LSTM base, ensamble de 3 semillas | 0.01101 | 0.01091 |

Predicciones de test: `outputs/submission.csv` (generado en el notebook 05).

## Estructura

```
Entrega_Lab2/
├── INFORME.md                informe del laboratorio
├── README.md
├── configs/                  un YAML por experimento (modelo + entrenamiento + resultado medido)
│   ├── 01_final/             final_cd.yaml → MODELO FINAL
│   ├── 02_base/              LSTM base · Adapformer k = 4 (top-k adaptativo del paper)
│   ├── 03_ablacion/          k = 1, sin ACE (k = 4 y k = 12), sin pérdida auxiliar
│   ├── 04_mejoras_descartadas/   pérdida híbrida, token de cuenca, EMA, embedding MLP, r = 64
│   └── 05_exploracion/       primera exploración (una semilla)
├── src/
│   ├── datos.py              lectura de los H5, log1p + z-score, caché float16, carga a GPU
│   ├── modelos.py            Adapformer (RevIN, SimBlock, ACE, ACF) + LSTM base
│   ├── entrenar.py           bucle de entrenamiento (línea de comandos o YAML)
│   ├── metricas.py           MSE, MAE, NSE/KGE por cuenca, por horizonte y por régimen
│   ├── resultados.py         tablas, ensambles, prueba de Wilcoxon
│   ├── hacer_configs.py      regenera configs/
│   └── fase*.sh, _comun.sh   experimentos en el orden en que se corrieron
├── notebooks/                00–05, ejecutados
├── figuras/                  figuras del informe
├── referencias_estudio1/     métricas de la versión inicial (k = 4, una semilla), usadas como comparación
└── outputs/
    ├── submission.csv        predicciones de test
    ├── stats.json            media y desviación por canal (train)
    └── <corrida>/            resumen.json y log.csv de cada corrida;
                              best.pt y predicciones sólo en final_cd_s* y lstm_s*
```

## Ejecución

Entorno: Python 3.12, PyTorch ≥ 2.4 con CUDA, h5py, numpy, pandas, scipy, matplotlib, pyyaml, jupyter y nbconvert.
Primero coloque el dataset (sección anterior). Desde la carpeta `Entrega_Lab2/`:

```bash
python src/datos.py                                                       # 1) caché float16 (~3 min, una vez)
python src/entrenar.py --config configs/01_final/final_cd.yaml --seed 42  # 2) una corrida (~2 min)
bash src/fase1_mejoras.sh       # exploración, 1 semilla
bash src/fase1b_semillas.sh     # pérdidas y token de cuenca, 3 semillas
bash src/fase1c_ema.sh          # EMA, 3 semillas
bash src/fase2_final.sh         # LSTM base, modelo final k = 12, ablación de componentes
bash src/fase3_cd.sh            # ablación de ACE sobre el modelo final
bash src/fase4_embedding.sh     # embedding MLP y r = 64
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
```

- Los scripts saltan las corridas que ya existen en `outputs/`.
- Para cambiar un hiperparámetro sin editar el YAML: `--set k=8 r=64`.
- Tiempo total medido: ~1.5 h en una RTX 4060 Laptop.
- `cache/` (2.6 GB) no se incluye: es el primer paso antes de re-ejecutar cualquier notebook o entrenamiento.
- Para re-ejecutar el notebook 04 hacen falta las predicciones de todas las corridas, que se obtienen con
  `bash src/fase*.sh`.
