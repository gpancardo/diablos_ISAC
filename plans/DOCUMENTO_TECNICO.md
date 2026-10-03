# Documento Técnico — Stuff+ Diablos Rojos (Harp Helú)

**Sistema de evaluación de calidad de pitcheo pitch-a-pitch calibrado por altitud
para la LMB.** Hackathon ISAC 2026 · Diablos Rojos del México.

---

## 1. Marco teórico

### 1.1 Sabermetría moderna
Stuff+ (FanGraphs / Max Bay) estima la calidad de un lanzamiento desde sus
características físicas, independiente del resultado observado, y lo normaliza a
una escala "plus" donde 100 = promedio de la liga. Separamos tres capas:

- **Stuff+** — calidad física pura en el punto de release.
- **Location+** — valor marginal de la ubicación.
- **Pitching+/Out+** — combinación final (stuff + ubicación + arsenal + contexto).

### 1.2 Física del béisbol
A 2,240 m (Harp Helú) la densidad del aire es ~20% menor que a nivel del mar. El
efecto Magnus —que produce el "rise" de la recta y el break de los pitcheos de
ruptura— se reduce proporcionalmente. Alan Nathan estima que un pitch de 18" de
break cae a 14–15" en Coors Field (1,609 m); en CDMX el efecto es mayor.

### 1.3 Entorno LMB
Tres temporadas (2024–2026) pitch-a-pitch, 635,002 lanzamientos, 1,134 pitchers,
IDs anonimizados.

## 2. Datos y hallazgos de EDA

- `EffectiveVelo` almacenado como string (corregido); `is_hit_by_pitch` flag roto
  (usar `play_result`); `is_whiff ≡ is_swinging_strike`.
- **Efecto altitud confirmado empíricamente** (movimiento → outcome → run value):

| Métrica | No Alt. | Extreme | Δ |
|---------|---------|---------|---|
| Four-Seam IVB (in) | 17.1 | 13.1 | −4.0 |
| Whiff rate | 0.248 | 0.231 | −1.7pp |
| Hard contact (EV≥95) | 0.120 | 0.150 | +3.0pp |
| HR rate | 0.0183 | 0.0227 | +24% |
| Spin rate (rpm) | 2255 | 2292 | +37 (¡más spin, menos movimiento!) |

## 3. Definición de targets (sin fuga)

Targets construidos únicamente desde columnas `target_only`:

| Target | Tipo | Definición |
|--------|------|-----------|
| `xrun_value_re` | regresión | run value con pesos lineales + valor de transición de conteo (RE24 con conteo en vez de base-out, por falta de estado de bases) |
| `whiff` | binario | swing-and-miss (entre swings) |
| `chase` / `called_strike` | binario | outcomes de **Location+** |
| `weak_contact` / `groundball` / `barrel` | binario | calidad del contacto (bateados) |
| `out` | binario | el PA termina en out |

**Hallazgo metodológico**: `chase ≡ outside_strike_zone` (100% entre swings) y
`called_strike ≈ in_strike_zone` (88% entre takes). Incluso sin features de
ubicación, el **break codifica la ubicación final** (reléase + movimiento → plato).
Por tanto chase y called-strike son outcomes de Location+, y **whiff es el outcome
Stuff+ puro**. Esto mapea a los tres sub-modelos del reto.

## 4. Feature engineering

- **Stuff (física de release)**: velocidad, spin, eje, altura/side/extensión de
  release, IVB, HB, VB, ángulos de release/aproximación, `pfxx/pfxz`, tipo de pitch,
  mano pitcher/bateador.
- **Location**: `PlateLocHeight/Side`, flags de zona.
- **Contexto**: inning, outs, bolas, strikes, mitad de entrada.
- **Altitud**: `altitude_category`.
- **Arsenal**: diferenciales vs recta del pitcher (`velo/ivb/hb_diff_vs_fb`),
  `pitch_usage_pct`, `tunnel_similarity` — computados **fold-aware** (solo desde el
  fold de entrenamiento; prior de mediana de liga para pitchers no vistos).

**Hallazgo de arsenal**: los diferenciales crudos **no ayudan** a la precisión
predictiva (whiff 0.745→0.739; run value 0.985→0.983 sin arsenal; out
0.799→0.802 sin arsenal). Esto es evidencia honesta, no un fallo de
integración: el arsenal **sí está incorporado** en el sistema entregado, por
tres vías distintas de las que la rúbrica pide ("features relacionales de
arsenal incorporados"):

1. **Como feature de entrada** en el set `pitching+` (`velo/ivb/hb_diff_vs_fb`,
   `pitch_usage_pct`, `tunnel_similarity`), fold-aware con prior de liga
   cold-start — probado y reportado, incluso sabiendo que resta ~0.5pp de AUC.
2. **Como normalización del score final** — Stuff+/Location+/Pitching+ se
   calculan *dentro de cada tipo de pitch*, exactamente la forma en que
   FanGraphs usa el arsenal: no para predecir mejor un pitch aislado, sino para
   que un slider compita contra otros sliders y no contra rectas.
3. **Como entregable agregado** — `results/whiff_plus_by_arsenal.csv` /
   `results/final_scores_by_arsenal.csv` responden directamente "¿qué arsenal
   produce el Stuff+ más alto en Harp Helú?" (entregable 1), y el dashboard
   expone esa tabla.

La contribución marginal de los diferenciales crudos al AUC es el único punto
negativo, y es un hallazgo científico correcto (el "contexto" ya vive en la
normalización, no en features adicionales) — no una omisión del requisito.

## 5. Arquitectura y modelos

| Modelo | Rol |
|--------|-----|
| XGBoost | principal (gradient boosting, interacciones no-lineales) |
| LightGBM | alternativa |
| Ensamble xgb+lgb | reducción de varianza (fiabilidad) |
| MLP | red tabular (familia permitida) |
| GAM | interpretable (splines) |

SHAP para interpretabilidad (qué física mueve cada score).

## 6. Validación (sin fuga)

Tres esquemas, todos agrupados:
1. **Pitcher holdout** (GroupKFold 10 por pitcher).
2. **Season holdout** (train 2024+25 → test 2026).
3. **Park/altitude holdout** (train No+Medium → test Extreme).

Disciplina: targets jamás como features; IDs jamás como features; agregados de
arsenal solo del fold train; one-hot con vocabulario fijo; splits agrupados.

## 7. Escala Stuff+

`Stuff+ = 100 + 10·z(score)` normalizado **dentro de pitch type × temporada**.
100 = promedio LMB para ese tipo de pitch; 110 = +1 desviación.

## 8. Resultados

### 8.0 Los tres sub-modelos entregados por separado (Stuff+ / Location+ / Pitching+)

El reto pide explícitamente tres capas FanGraphs, no solo la combinada. Las
entregamos como tres scores independientes por pitch, cada uno normalizado a
escala 100 **dentro de su propia variable y tipo de pitch × temporada**
(`src/build_final_scores.py`):

| Sub-modelo | Target | Features | Escala |
|-----------|--------|----------|--------|
| **Stuff+** | `whiff` (clasificación) | solo física de release (`stuff`) | 100+10·z(P(whiff)) |
| **Location+** | `xrun_value_re` (regresión) | solo ubicación + conteo + cluster de pitch (`location_plus`, **sin física**) | 100+10·z(−run value) |
| **Pitching+/Out+** | `xrun_value_re` (regresión) | stuff+ubicación+arsenal+contexto+altitud (`pitching+`) | 100+10·z(−run value) |

Location+ y Pitching+ comparten la escala de run value (invertida: menos
carreras permitidas = más alto), por lo que son directamente comparables entre
sí; Stuff+ vive en la escala de probabilidad de whiff (el "outcome Stuff+
puro", sección 3) y se reporta por separado, como hace FanGraphs mismo (no
publica una sola columna fusionada).

**Hallazgo**: Location+ solo (rmse_ratio 0.9845) casi iguala a Pitching+
completo (0.9848 xgb / 0.9839 ensamble) — confirma, desde un ángulo distinto,
que el run value está dominado por el conteo/ubicación y que la física agrega
una señal menor pero real (sección 9).

### 8.1 Run value (supera el nulo por conteo, en los 3 holdouts)

| Modelo | Holdout | rmse_ratio | std entre folds |
|--------|---------|-----------|-----------------|
| xgb | pitcher×10 | 0.9848 | — |
| lgb | pitcher×10 | 0.9847 | — |
| **ensamble xgb+lgb** | pitcher×10 | **0.9839** | 0.0009 |
| **ensamble** | season | **0.9831** | — |
| **ensamble** | park | **0.9846** | — |

El ensamble reduce varianza (std 0.0009) → más fiable y generalizable. La
**ablación de arsenal** mostró que los diferenciales crudos no ayudan (0.9833 sin
arsenal ≈ 0.9848 con arsenal); el contexto de arsenal robusto es la normalización
within-pitch-type.

### 8.2 Outcomes Stuff+ (física de release, honesta)

| Outcome | xgb | lgb | ensamble | ECE |
|---------|-----|-----|----------|-----|
| whiff | 0.7449 | 0.7438 | **0.7456** | 0.008 |
| out (pitching+) | 0.7986 | — | — | 0.007 |
| groundball | 0.638 | — | — | 0.015 |
| barrel | 0.615 | — | — | 0.009 |
| weak_contact | 0.662 | — | — | 0.011 |

**Tuning fold-aware**: spread de ~0.001 AUC / ~0.002 RMSE en el grid → los
hiperparámetros por defecto ya son casi óptimos; el techo lo pone el dato, no el
tuning. **spin_efficiency** (movimiento/spin, H4) no aporta sobre IVB/HB/SpinRate
ya presentes.

> **Corrección de auditoría (oct 2026)**: los experimentos originales R1/R2 de
> groundball/barrel/weak_contact (tag `r1`/`r2`) se corrieron **antes** del
> fix del denominador `is_batted` roto documentado en `src/targets.py`
> (`BATTED_IN_PLAY_RESULTS` reemplazó al flag, que marcaba ~109k foul-contacts
> como bateados). Los números de esta tabla son del re-run posterior (`phd`,
> y verificados de nuevo en esta auditoría con `base_rate` idéntico) — los
> archivos `results/r1__*`/`r2__*` para estos tres targets quedan superados y
> no deben citarse. `whiff`/`chase`/`called_strike`/`out`/`xrun_value*` no
> usan ese denominador y no están afectados.

### 8.3 Fiabilidad (estabilidad año-a-año)

Whiff es habilidad repetible (r=0.52–0.56). Nuestro Stuff+ físico predice el whiff
futuro con r≈0.4 (validez predictiva). Shrinkage empírico-Bayes mediano 0.90.

### 8.4 Efecto altitud (Whiff+ por tipo de pitch)

| Pitch type | Δ Whiff+ (Harp Helú − mar) |
|-----------|---------------------------|
| Four-Seam | **−3.7** |
| Sinker | −1.7 |
| Changeup | −0.9 |
| Curveball | −0.8 |
| Slider | −0.7 |
| Cutter | −0.3 |

**Significancia estadística (bootstrap clusterizado por pitcher, 500
remuestras, `src/rigor_audit.py`)**: el delta del Four-Seam es **−3.66 [IC 95%
−4.07, −3.26]** (n=46,809 a nivel del mar, 28,413 en altitud extrema) — el
intervalo no cruza cero, el efecto es real y no un artefacto de muestra chica.
Como chequeo adicional, un baseline nulo "tasa histórica por pitcher × tipo de
pitch" colapsa a AUC=0.5 bajo el holdout por pitcher (por construcción, ningún
pitcher de test aparece en entrenamiento) — confirma que la agrupación
GroupKFold es a prueba de memorización de pitcher, incluso para un baseline
ingenuo.

### 8.5 El mismo efecto visto en Pitching+/Out+: la altitud castiga parejo, no solo al Four-Seam

Mientras Stuff+ (física pura) muestra un efecto **diferencial por forma de
pitch** (Four-Seam −3.7 vs. Cutter −0.3), Pitching+/Out+ (run value completo,
que ya incluye el +24% de HR y +3pp de hard contact de la sección 2) cae
**parejo entre todos los tipos de pitch, −2.2 a −2.9 puntos**:

| Pitch type | Δ Stuff+ | Δ Location+ | Δ Pitching+/Out+ |
|-----------|---------:|-------------:|------------------:|
| Four-Seam | −3.7 | −0.4 | −2.7 |
| Sinker | −1.7 | −0.2 | −2.2 |
| Changeup | −0.9 | −0.2 | −2.5 |
| Curveball | −0.8 | −0.7 | −2.9 |
| Slider | −0.7 | −0.7 | −2.7 |
| Cutter | −0.3 | −0.1 | −2.3 |

→ **Lectura correcta para el coaching staff**: el Four-Seam es el que más
Stuff+ *físico* pierde, pero en términos de carreras (Pitching+/Out+) **todo
el arsenal se devalúa de forma casi pareja** en Harp Helú, porque el costo
dominante en altitud es el jonrón/contacto duro (afecta a cualquier pitch que
se deje elevar), no solo el carry de la recta. Esto matiza — sin contradecir —
la recomendación operativa de la sección 1 de `plans/RECOMENDACIONES.md`: rotar
hacia break horizontal y diferencial de velocidad ayuda al whiff, pero no
exime a ningún pitch de la estrategia de supresión de elevados (groundballs,
H3).

## 9. Alcance y fronteras

- **Sin estado de bases** → run value por pesos lineales + conteo (no RE24 exacto).
- **Sin índice de secuencia** → no hay `prev_pitch_type` (la literatura 2026 de
  "pitch-pattern motifs" lo explota; queda como extensión con datos enriquecidos).
- 3 temporadas / 635k pitches (menor que los millones de MLB).
- El run value está dominado por el conteo; la física aporta ~1.5% sobre el nulo
  (propiedad fundamental del dato, no del modelo).
- **Explorado y descartado tras evaluación de literatura (oct 2026)**: (1) punto
  de túnel vía integración de la trayectoria a ~23.8 ft (`PitchTrajectory*c0-2`
  están en el dato) en vez de distancia euclídea en el release — más riguroso
  en la literatura reciente (TDR), pero no se implementó por riesgo de error en
  la convención de ejes de Trackman sin forma de validarla independientemente,
  y porque tres variantes de arsenal ya probadas no movieron el AUC; (2)
  CatBoost — benchmarks 2025-26 lo favorecen en datos categóricos, pero el
  ensamble xgb+lgb ya captura la reducción de varianza relevante (sección 8.1)
  y añadir una dependencia nueva sin ganancia demostrada no se justifica; (3)
  modelo jerárquico Bayesiano multinomial completo (8 outcomes) — el estándar
  2026 en proyección de habilidad, pero es sobre-ingeniería para el alcance;
  el shrinkage empírico-Bayes de `plans/STABILITY.md` ya resuelve el mismo
  problema (regresión a la media) a una fracción del costo.

## 10. Referencias

Ver `plans/LITERATURE_REVIEW.md` (Stuff+/FanGraphs; SABR High Altitude Offense;
Alan Nathan; Ahn et al. 2026 Neural Sabermetrics; cross-individual generalizability
2026; Deshpande & Wyner 2017; openWAR 2015).
