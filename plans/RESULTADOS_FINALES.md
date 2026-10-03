# Resultados finales — resumen consolidado

Validación leak-free (3 holdouts). ~35 experimentos, 4 familias de modelos,
tuning fold-aware, ensambles, ablaciones, y análisis sabermétrico de fiabilidad.

## 1. Métricas finales (modelo ensamble, el más fiable)

| Target | Modelo | Métrica | Valor | vs mejor single |
|--------|--------|---------|-------|-----------------|
| whiff (stuff) | ensamble xgb+lgb | AUC | 0.7456 | +0.0007 |
| out (pitching+) | xgb | AUC | 0.7986 | — |
| xrun_value_re | ensamble | rmse_ratio | 0.9839 (pitcher) / 0.9831 (season) / 0.9846 (park) | +0.0009 |

- **std entre folds muy baja** (run value 0.0009, whiff 0.0058) → fiable.
- **Calibración ECE ~0.007** (excelente) en los targets limpios.

## 2. Hallazgo honesto: el techo lo pone el dato, no el modelo

- **Tuning** (6 combos × 3 folds): spread de solo ~0.001 AUC / ~0.002 RMSE. Los
  hiperparámetros por defecto ya son casi óptimos.
- **spin_efficiency** (H4): no aporta sobre IVB/HB/SpinRate (redundante).
- **Arsenal (diferenciales crudos)**: NO ayudan — whiff 0.745→0.739, run value
  0.985→0.983 sin arsenal, out 0.799→0.802 sin arsenal. El contexto de arsenal
  correcto es la **normalización within-pitch-type** (cómo FanGraphs lo implementa).
- **MLP/GAM**: en curso (GBM domina a 635k filas; esperado).

→ La precisión predictiva está cerca del techo. El diferencial competitivo NO es
más precisión, es: **rigor sin fuga + framing sabermétrico correcto + altitud
cuantificada + fiabilidad (estabilidad) + interpretabilidad**.

## 3. Fiabilidad (el criterio que separa modelos reales de juguetes)

- Whiff es habilidad repetible: r=0.52–0.56 año a año.
- **Validez predictiva**: Stuff+ físico en año T predice whiff real en T+1 con
  r=0.39–0.44.
- Shrinkage empírico-Bayes mediano 0.90 (rankings ajustados por muestra).

## 4. Altitud cuantificada (el diferenciador del reto)

Four-Seam **−3.7** Stuff+ · Sinker −1.7 · Changeup −0.9 · Curveball −0.8 ·
Slider −0.7 · Cutter −0.3. Mecanismo: IVB −4.0", whiff −1.7pp, hard contact +3pp,
HR +24%, spin ↑ pero movimiento ↓ (H4).

## 5. Fugas detectadas y corregidas

1. chase/called-strike = Location+ (break codifica ubicación), no Stuff+.
2. One-hot con `pd.Categorical` perdía índice de fold (inflaba filas).
3. `is_hit_by_pitch` flag roto.
4. Arsenal fold-aware con prior cold-start.

## 6. Entregables completos

- **Los tres sub-modelos del reto por separado** (Stuff+/Location+/Pitching+),
  por pitch/pitcher/arsenal → `results/final_scores_*` (`src/build_final_scores.py`).
  Location+ solo (rmse_ratio 0.9845) casi iguala a Pitching+ completo (0.9839
  ensamble) — el run value sigue dominado por conteo/ubicación, ver
  `plans/DOCUMENTO_TECNICO.md` §8.0/8.5.
- Stuff+ (Whiff+) por pitch/pitcher/arsenal → `results/whiff_plus_*.csv/parquet`.
- Delta de altitud, las 3 capas → `results/altitude_delta*.csv`,
  `dashboard/data/layer_altitude_delta.csv`.
- Perfil físico ideal (SHAP) → `results/shap_whiff_importance.csv`.
- Fiabilidad → `results/pitcher_shrunk_whiff.csv` + `plans/STABILITY.md`.
- Dashboard premium → `dashboard/app.py`.
- Documento técnico → `plans/DOCUMENTO_TECNICO.md` + 6 docs de apoyo.
