# Plan de Experimentos — Stuff+ Diablos Rojos (Harp Helú)

Hackathon ISAC 2026 · Liga Mexicana de Béisbol · ~635k lanzamientos (2024–2026)

## 1. Objetivo y definición del problema

Construir un **Stuff+ Model** pitch-a-pitch que estime la calidad esperada de cada
lanzamiento desde sus propiedades físicas, su ubicación, su relación con el arsenal
del pitcher y el entorno atmosférico (altitud CDMX ≈ 2,240 m).

Escala objetivo: `Stuff+ = 100 + 10 · z(score)`, donde 100 = promedio LMB ajustado
por temporada/estadio.

Tres sub-modelos (marco FanGraphs Stuff+ / Location+ / Pitching+):

| Capa | Inputs | Pregunta |
|------|--------|----------|
| **Stuff+** | física de release (velo, spin, break, release) | ¿qué tan bueno es el pitch por su física? |
| **Location+** | plato, zona, count, lado del bateador | ¿qué tan bien ubicado está? |
| **Pitching+/Out+** | stuff + location + arsenal + contexto + altitud | probabilidad de outcomes + run value |

## 2. Definición de targets (sin fuga)

Los targets se derivan **únicamente** de columnas `target_only` del diccionario. No
existe ruta de fuga outcome→feature porque las columnas de resultado jamás entran al
conjunto de features.

| Target | Tipo | Subconjunto | Definición |
|--------|------|-------------|-----------|
| `xrun_value` | regresión | todos | pesos lineales del evento terminal (`play_result`) |
| `whiff` | binario | swings | `is_whiff` (swing y falla) |
| `chase` | binario | swings | `swung_outside_strike_zone` |
| `called_strike` | binario | takes | `is_called_strike` |
| `out` | binario | todos | `OutsOnPlay >= 1` |
| `weak_contact` | binario | bateados | `ExitSpeed < 80 mph` |
| `groundball` | binario | bateados | `hit_type == GroundBall` |
| `barrel` | binario | bateados | `ExitSpeed≥95 & 20°≤Angle≤40°` (proxy) |

**Limitación de datos documentada:** no hay estado de bases (runners_on_base), ni
identificador de plate-appearance, ni secuencia de pitch. Por tanto no es computable
el RE24 exacto. Usamos *pesos lineales por evento* (0 para pitches no terminales) como
proxy de run value — es la aproximación clásica y no requiere base-out state. El
baseline nulo del certamen ("xRunValue promedio por conteo") se reproduce exactamente.

## 3. Feature engineering

### 3.1 Conjuntos de features (por sub-modelo)

- **stuff** (física de release): `RelSpeed, EffectiveVelo, SpinRate, SpinAxis,
  RelHeight, RelSide, Extension, InducedVertBreak, HorzBreak, VertBreak,
  VertRelAngle, HorzRelAngle, VertApprAngle, HorzApprAngle, SpeedDrop, pfxx, pfxz`.
  *Se descartan `Tilt` (redundante con SpinAxis) y `ZoneSpeed` (corr .986 con
  RelSpeed) para evitar colinealidad.*
- **location**: `PlateLocHeight, PlateLocSide, in_strike_zone, outside_strike_zone`.
- **contexto**: `Inning, Outs, Balls, Strikes`.
- **categóricas**: `AutoPitchType, PitcherThrows, BatterSide, altitude_category,
  Top/Bottom` (one-hot con vocabulario fijo).

### 3.2 Features de arsenal (el diferenciador — "arsenal como contexto")

Computadas **siempre desde el fold de entrenamiento**; los pitchers no vistos
reciben prior de mediana de liga (cold-start). Esto es lo que hace robusta la
generalización a pitchers nuevos (25% de la nota).

- `velo_diff_vs_fb` = RelSpeed − media de la recta (Four-Seam+Sinker) del pitcher
- `ivb_diff_vs_fb`, `hb_diff_vs_fb` = diferencial de movimiento vs su recta
- `pitch_usage_pct` = frecuencia de ese tipo de pitch en el repertorio del pitcher
- `tunnel_similarity` = −distancia euclídea del punto de release a su recta

### 3.3 Altitud

`altitude_category` entra como feature categórica (No/Medium/Extreme Altitude).
El **delta Stuff+ por altitud** se cuantifica post-hoc: mismo pitch predicho con
`altitude_category = Extreme` vs `No Altitude` (análisis contrafactual explícito
que respalda H1–H4).

## 4. Arquitectura y modelos permitidos

| Modelo | Rol | Notas |
|--------|-----|-------|
| **XGBoost** | principal (gradient boosting) | interacciones no-lineales; SHAP para el coach |
| **LightGBM** | alternativa | mismo marco, comparación de robustez |
| **GAM (pyGAM)** | interpretable | splines en velo/spin/break — sanity check físico |

Modelos adicionales permitidos (fase 2, opcional): Bayesiano jerárquico (shrinkage
de medias por pitcher), red tabular, Causal Forest (capa táctica "slider vs cambio").

## 5. Validación sin fuga (tres esquemas = 25% de la nota)

1. **Pitcher holdout** — `GroupKFold` (10) por `pitcher_anon_id`. El modelo debe
   generalizar a pitchers jamás vistos. Features de arsenal vía prior de liga.
2. **Season holdout** — train 2024+2025 → test 2026 (cronológico, sin fuga temporal).
3. **Park/altitude holdout** — train No+Medium → test Extreme Altitude. Aísla el
   efecto altitud y calibra por estadio.

**Disciplina de fuga (checklist):**
- targets jamás como features (colisiones `target_only` excluidas por diseño).
- `pitcher_anon_id`/`game_anon_id`/`batter_anon_id` jamás como features (memorización).
- agregados de arsenal solo desde el fold train; prior de liga para cold-start.
- encoding one-hot con vocabulario fijo (sin fuga de niveles raros).
- splits agrupados (nunca el mismo pitcher en train y test del mismo fold).

## 6. Métricas y baseline

- Regresión (`xrun_value`): RMSE, MAE, bias; `rmse_ratio = RMSE / RMSE_null`.
- Clasificación: AUC, Brier, log-loss; baseline nulo = media por conteo.
- **Superar el baseline nulo (media por conteo)** es el piso; la señal fuerte vive en
  los targets secundarios (whiff AUC ≈ 0.75 en smoke test).

## 7. Matriz de experimentos (Ronda 1)

| # | target | modelo | features | holdout |
|---|--------|--------|----------|---------|
| 1 | xrun_value | xgb | pitching+ | pitcher×10 |
| 2 | xrun_value | lgb | pitching+ | pitcher×10 |
| 3 | xrun_value | xgb | stuff | pitcher×10 |
| 4 | xrun_value | xgb | stuff+location | pitcher×10 |
| 5 | xrun_value | xgb | pitching+ | season |
| 6 | xrun_value | xgb | pitching+ | park |
| 7–10 | whiff/chase/called_strike/out | xgb | pitching+ | pitcher×10 |
| 11–13 | weak_contact/groundball/barrel | xgb | pitching+ | pitcher×10 |
| 14 | xrun_value | gam | stuff | pitcher×5 |

Ronda 2 (tras leer R1): tuning de hiperparámetros, composición del score Stuff+,
análisis SHAP, delta de altitud contrafactual, y jerárquico bayesiano.

Ronda 3 (post-auditoría, oct 2026): sub-modelo **Location+** explícito
(`xrun_value_re` con feature set `location_plus` = solo ubicación+conteo+
cluster de pitch, sin física) para entregar las tres capas del reto por
separado (`src/build_final_scores.py`); sweep chico de feature engineering
sobre whiff (`stuff_v2`: VAA/HAA normalizado por velocidad, |HorzBreak|) y de
hiperparámetros (árboles↑, learning rate↓) — ver `plans/STUFF_PLUS_RESULTS.md`
§5b para los deltas de AUC.

## 8. Entregables → artefactos del repo

1. Scores por pitch/pitcher/arsenal (Stuff+, Location+, Pitching+) →
   `src/build_final_scores.py` (Ronda 3)
2. Perfil físico ideal en altitud → SHAP + delta altitud → `plans/`
3. Recomendaciones operativas → `plans/RECOMENDACIONES.md`
4. Dashboard simulador → `dashboard/` (Streamlit, Ronda 2)
5. Documento técnico 6–10 pp → `plans/`
