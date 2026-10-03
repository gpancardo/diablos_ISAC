# Reporte de EDA — Stuff+ Diablos Rojos

Dataset: `stuff_model_df.parquet` — **635,002 lanzamientos × 84 columnas**, LMB
2024–2026, sin PII, IDs anonimizados.

## 1. Estructura y cardinalidad

- Temporadas: 2024 (243,254) · 2025 (256,144) · 2026 (135,604).
- 1,134 pitchers · 864 bateadores · 196 catchers · 2,127 juegos.
- `PitchUID` único (0 duplicados).

## 2. Calidad de datos

| Hallazgo | Detalle | Acción |
|----------|---------|--------|
| `EffectiveVelo` es **string** | 100% convertible a float (38–109 mph) | `to_numeric` |
| `is_hit_by_pitch` **roto** | flag ≈ 0.0 pero `play_result==HitByPitch` = 2,258 | usar `play_result`, no el flag |
| `is_whiff` == `is_swinging_strike` | redundantes (corr 1.0) | usar uno solo |
| Missing relevante | batted-ball (ExitSpeed/Angle/Distance ~71–78%) solo en contacto | natural; subsetear |
| Missing menor | SpinRate/SpinAxis/break ~0.11%; altitude_category 0.48% | imputación/sentinel |
| `NeutralPlay` (199k) | pitch **no terminal** (PA continúa: fouls, called strikes no-K) | clave para targets |

## 3. Taxonomía de outcomes (targets)

`play_result` (16 valores) es el outcome terminal canónico. `PitchCall` cruza a
`play_result` de forma limpia:
- `BallCalled` + `BallinDirt` → `Walk` (si bola 4) o no-terminal.
- `StrikeCalled` → `Strikeout` (si strike 3) o `NeutralPlay`.
- `StrikeSwinging` → `Strikeout` (si strike 3) o whiff no-terminal.
- `FoulBall*` → `NeutralPlay`.
- `InPlay` → Single/Double/Triple/HomeRun/Out/Error/FieldersChoice/Sacrifice.

→ **`NeutralPlay` = pitch que mantiene vivo el PA** (no terminal). Sustenta la
construcción de targets sin fuga.

## 4. Efecto altitud (hipótesis H1–H4 confirmadas empíricamente)

Movimiento inducido **cae monotónicamente** con la altitud:

| Métrica | No Alt. | Medium | Extreme | Δ No→Extreme |
|---------|---------|--------|---------|--------------|
| InducedVertBreak (in) | 9.01 | 7.47 | 6.87 | **−2.14** |
| HorzBreak (in) | 3.63 | 2.06 | 2.26 | −1.37 |
| VertBreak (in) | −28.25 | −29.12 | −29.44 | −1.19 |

Por tipo de pitch (Four-Seam): IVB **17.1" → 13.1"** (−4.0") en Extreme Altitude.
Curva y slider pierden break; el changeup (dependiente de diferencial de velo, no de
Magnus) cae menos — **H2 respaldada**.

**H4 confirmada:** `SpinRate` es *mayor* en altitud (2292 vs 2255 rpm) pero produce
*menos* movimiento → "shape over label": el spin no se traduce en break real.

**H3 (groundball):** por cuantificar en Ronda 2 (rate de groundball × altitud × valor).

## 5. Colinealidad (selección de features)

- `pfxz` ↔ InducedVertBreak corr **0.997**; `VertBreak` ↔ IVB 0.947.
- `ZoneSpeed`/`EffectiveVelo`/`ZoneTime` ↔ RelSpeed corr 0.98+.
→ Se retienen features físicamente distintas; se descartan redundantes.

## 6. Base rates de targets

whiff 11.3% (24.1% de swings) · chase 22.2% · called_strike 15.4% · HR 0.73% ·
strikeout 4.8% · walk 2.5% · contacto duro (≥95mph) por medir en Ronda 2.

## 7. Auditoría de fuga

Diccionario coherente: 26 `target_only`, 41 `stuff_feature`, 4 `location_plus_only`,
8 `context_only`, 4 `grouping_only`, 1 `id_only`. **Sin inconsistencia** entre
`feature_role` y `use_as_model_feature`. Ver `src/features.py` para el cumplimiento
estricto.
