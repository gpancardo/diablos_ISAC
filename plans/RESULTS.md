# Resultados de experimentos — Stuff+ Diablos Rojos

Validación leak-free con 3 esquemas de holdout. Todos los targets se construyen
solo desde columnas `target_only`; features solo desde `stuff_feature` /
`location_plus_only` / `context_only`; arsenal computado fold-aware.

## 1. Run value — supera el baseline por conteo en los 3 holdouts

| Target | Modelo | Features | Holdout | RMSE | rmse_ratio |
|--------|--------|----------|---------|------|-----------|
| xrun_value_re (count-aware) | xgb | pitching+ | pitcher×10 | 0.2177 | **0.9848** |
| xrun_value_re | xgb | pitching+ | season (24+25→26) | 0.2160 | **0.9847** |
| xrun_value_re | xgb | pitching+ | park (no+med→extreme) | 0.2259 | **0.9857** |
| xrun_value_re | lgb | pitching+ | pitcher×10 | 0.2176 | **0.9847** |
| xrun_value (lineal simple) | xgb | pitching+ | pitcher×10 | 0.2155 | 0.9866 |

→ Beat consistente de ~1.5% sobre el nulo, sin fuga, generalizando a pitchers,
temporadas y estadios no vistos. El target *count-aware* (RE24 con conteo en vez
de base-out) es marginalmente mejor y más principista.

## 2. Outcomes de Stuff+ (física de release, honesta — sin ubicación)

| Outcome | AUC | Base rate | Interpretación |
|---------|-----|-----------|----------------|
| **whiff** | **0.745** | 0.241 | señal Stuff+ más fuerte y limpia |
| groundball | 0.638 | 0.424 | calidad de contacto |
| barrel | 0.615 | 0.062 | contacto duro (proxy) |
| weak_contact | 0.662 | 0.320 | contacto débil |

*(Números corregidos oct 2026: los valores originales de esta tabla usaban el
denominador `is_batted` roto, ver nota de auditoría en `DOCUMENTO_TECNICO.md`
§8.2. El AUC no cambia por el fix —cambia el conjunto de filas evaluado— pero
el `base_rate` sí, y por honestidad usamos siempre el re-run posterior.)*
| **out** (pitching+) | **0.799** | 0.166 | out en el PA |

**Calibración**: ECE ~0.007 en los targets limpios (excelente).

## 3. Hallazgo metodológico clave: chase y called-strike son Location+, no Stuff+

- `chase` (swing fuera de zona) con features de *ubicación* → AUC **1.00**
  (fuga por definición: `swung_outside_strike_zone ≡ outside_strike_zone` entre swings).
- `called_strike` con ubicación → AUC 0.97 (88% definicional).
- **Incluso sin features de ubicación**, chase (0.98) y called_strike (0.96)
  siguen altísimos: el **break codifica la ubicación final del pitch** (reléase +
  movimiento → plato → zona). Es física, no fuga.

→ Conclusión correcta de sabermetría: **chase y called-strike pertenecen a
Location+**; **whiff es el outcome Stuff+ puro**. Esto mapea exactamente a los tres
sub-modelos del reto.

## 4. Ablaciones (whiff, el target con más señal)

| Variante | AUC | Δ |
|----------|-----|---|
| stuff (física) | 0.7449 | — |
| stuff + location | 0.7524 | +0.0075 (ubicación aporta poco a whiff) |
| stuff + arsenal | 0.7390 | **−0.006 (el arsenal NO ayuda a whiff)** |
| lgb (stuff) | 0.7438 | xgb ≈ lgb |

→ El whiff lo determina la física absoluta (velo, spin, break), no los
diferenciales de arsenal. Hallazgo honesto a reportar.

## 5. Efecto altitud (movimiento → outcome → run value)

| Métrica | No Alt. | Extreme | Δ |
|---------|---------|---------|---|
| Four-Seam IVB (in) | 17.1 | 13.1 | −4.0 |
| Whiff rate | 0.248 | 0.231 | −1.7pp |
| Four-Seam whiff | 0.214 | 0.181 | −3.3pp |
| Hard contact (EV≥95) | 0.120 | 0.150 | +3.0pp |
| HR rate | 0.0183 | 0.0227 | +24% |
| Spin rate (rpm) | 2255 | 2292 | +37 (¡más spin, menos movimiento!) |

H1 (menos movimiento), H2 (changeup/breaking ganan valor relativo), H3 (groundball
vale más) y H4 (shape over label) confirmadas con datos LMB. Ver
`plans/ALTITUDE_FINDINGS.md`.

## 6. Fuga detectada y corregida

1. `chase`/`called_strike` con features de ubicación → leak por definición (R1,
   AUC 1.0/0.97). Corregido: modelar outcomes de stuff con física sola.
2. One-hot encoding: `pd.get_dummies(pd.Categorical(...))` perdía el índice de
   fold y duplicaba filas en el concat → corregido envolviendo en Series.
3. `is_hit_by_pitch` flag roto (≈0 vs 2,258 reales) → usar `play_result`.

## 7. Artefactos generados

- `results/*.json` — métricas por experimento (R1 + R2).
- `results/*.oof.parquet` — predicciones out-of-fold (para Stuff+ y delta).
- `plans/` — EXPERIMENT_PLAN, EDA_REPORT, LITERATURE_REVIEW, ALTITUDE_FINDINGS.
- `src/` — eda, targets, features, train, run_experiments, summarize,
  stuff_score, altitude_delta, shap_analysis, altitude_analysis.
