# Stuff+ scores (Whiff+) — cuantificación del efecto altitud

`Whiff+ = 100 + 10·z(P(whiff))` normalizado **dentro de pitch type × temporada**.
Fuente: predicciones out-of-fold del modelo whiff con física de release (AUC 0.745).

## 1. Delta Whiff+ por pitch type al mover a Harp Helú (la pregunta del reto)

| Pitch type | No Altitude | Extreme Altitude | Δ |
|-----------|-------------|------------------|---|
| Four-Seam | 101.8 | 98.1 | **−3.7** |
| Sinker | 100.8 | 99.1 | −1.7 |
| Changeup | 100.3 | 99.4 | −0.9 |
| Curveball | 100.4 | 99.6 | −0.8 |
| Slider | 100.4 | 99.7 | −0.7 |
| Cutter | 100.1 | 99.8 | −0.3 |

→ El Four-Seam (que depende del "rise" inducido por Magnus) pierde **3.7 puntos** de
Stuff+ en altitud; los pitches de ruptura y el changeup (engaño por diferencial de
velocidad) pierden **<1 punto**. Confirma H1/H2 con la escala propia del modelo.

## 2. Media Whiff+ por altitud

No Altitude 100.9 → Medium 99.4 → Extreme 99.1 (caída ~1.8 puntos por menor whiff).

## 3. Top pitchers por Whiff+ (n ≥ 200 swings)

pitcher_00643 (110.5), pitcher_01103 (108.8), pitcher_00754 (108.0) …
El club puede rankear su staff por capacidad de inducir whiff ajustada a su arsenal.

## 4. Perfil físico ideal (SHAP sobre P(whiff))

Lo que más induce whiff (importancia |SHAP| y dirección):
- **VertApprAngle** (más plano = más whiff) — la feature #1.
- **InducedVertBreak ↑** (más "rise" = más whiff) — y es justo lo que la altitud mata.
- **RelSpeed ↑**, **HorzBreak ↑**, **RelHeight ↑**, **VertRelAngle ↑**.
- **SpinRate** aporta poco relativo al movimiento observado (H4: shape over label).

→ El perfil que maximiza Stuff+ en CDMX: **velocidad alta + break horizontal +
release alto + approach plano**, y aceptar que el IVB se degrada en altitud (por
eso conviene rotar hacia sliders/curvas/changeups cuyo engaño no depende de Magnus).

## 5b. Sweep de mejoras chicas (oct 2026) — ¿hay agua en las piedras?

Cada palanca se probó **aislada** (una variable a la vez, mismo baseline) para
poder atribuir el efecto, siguiendo la literatura de pitch quality models
(Driveline VAA research, FanGraphs Stuff+ v2, `jakeyoung1/pitchquality`) y de
gradient boosting en problemas desbalanceados:

| Variante | Target | AUC | Δ vs baseline |
|----------|--------|-----|----------------|
| baseline (`stuff_clean`, physics de-correlacionada) | whiff | 0.7425 | — |
| + VAA/HAA ÷ velocidad + \|HorzBreak\| (`stuff_v2`) | whiff | 0.7426 | **+0.0001 (ruido, std≈0.006)** |
| 1200 árboles, lr 0.03, depth 7 (vs 600/0.05/6) | whiff | 0.7424 | **−0.0001 (ruido, 2× más lento de entrenar)** |
| `scale_pos_weight≈26` (target más desbalanceado, 6.2% positivos) | barrel | 0.6120 | **−0.0028, y ECE se dispara 0.009→0.392** |

**Por qué VAA/HAA normalizado no ayudó (literatura)**: la investigación de
Driveline muestra que la VAA de la recta de 4 costuras depende **únicamente**
de la altura de release y la ubicación del pitch, no de la velocidad ni del
movimiento — por lo que dividir por `RelSpeed` no aísla nada nuevo que XGBoost
no pueda ya separar con un split en la variable cruda + `RelSpeed` como
feature independiente (las interacciones ya las captura el árbol). Normalizar
a mano una variable que un GBM ya puede recombinar libremente rara vez ayuda;
sí ayudaría en un modelo lineal/GAM.

**Conclusión honesta**: las tres primeras palancas no movieron la aguja más
allá del ruido entre folds. Confirma, desde un ángulo distinto, el hallazgo de
R2: el techo lo pone la resolución física del dato (Trackman a 635k pitches),
no el feature engineering ni los hiperparámetros, para el target que *ya*
tiene señal fuerte (whiff). `scale_pos_weight` en barrel (el target más
desbalanceado) también falló: AUC prácticamente igual (−0.003, dentro del
ruido) pero la calibración se destruye (ECE 0.009→0.39) — técnica descartada,
consistente con la literatura (reponderar la clase cambia el objetivo de
optimización sin mejorar el ranking en un problema ya bien separado por
árboles). La ganancia competitiva real sigue siendo rigor + framing + altitud
cuantificada + las tres capas Stuff+/Location+/Pitching+ (sección 8.0 del
documento técnico) — no otro decimal de AUC.

**Quinta palanca (literatura "Exit Velocity Over Expected")**: en vez de
clasificar binario, regresar `ExitSpeed` continuo (`src/exp_ev_regression.py`,
mismo esquema pitcher-holdout 10-fold, mismas features `stuff`) y derivar el
umbral después. Resultado **mixto y explicable**:

| Target | AUC directo (clasificación) | AUC derivado de regresión EV | Δ |
|--------|------------------------------|-------------------------------|---|
| weak_contact (`ExitSpeed<80`) | 0.6624 | **0.6744** | **+0.012** |
| barrel (`EV≥95 & 20°≤ángulo≤40°`) | 0.6148 | 0.6042 | −0.011 |

weak_contact es literalmente un umbral de EV, así que regresar el valor
continuo (RMSE ratio 0.937 vs el nulo, Spearman 0.325) conserva señal que la
clasificación binaria tira. barrel además depende del ángulo de lanzamiento,
que la regresión de solo EV no ve, así que pierde frente al clasificador
directo que sí usa `ExitSpeed`+`Angle` en el target. **Ganancia real para
weak_contact, no generalizable a barrel.**

**Hallazgo colateral de esta auditoría** (no buscado, encontrado al correr el
baseline de `barrel` de nuevo para comparar limpio): los resultados R1/R2 de
`groundball`/`barrel`/`weak_contact` eran de **antes** del fix del denominador
`is_batted` en `targets.py` y quedaron obsoletos sin que se regeneraran los
documentos que los citaban. Corregido en `DOCUMENTO_TECNICO.md` §8.2 y
`RESULTS.md` — el AUC correcto y vigente es el de los archivos `phd__*`
(groundball 0.638, barrel 0.615, weak_contact 0.662), reproducido de nuevo en
esta sesión (`results/fe4base__barrel__xgb__stuff__pitcher.json`).

## 5. Artefactos

- `results/whiff_plus_pitch.parquet` — Whiff+ por pitch.
- `results/whiff_plus_by_pitcher.csv` — ranking de pitchers.
- `results/whiff_plus_by_arsenal.csv` — Whiff+ por arsenal × altitud.
- `results/altitude_delta_by_pitchtype.csv` — delta contrafactual de run value.
- `results/shap_whiff_importance.csv` — importancia SHAP.
