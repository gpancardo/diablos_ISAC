# Hallazgos empíricos — Efecto altitud (movimiento → outcome → run value)

Análisis de datos crudos (sin modelo). Todo por `altitude_category`, controlando
por tipo de pitch.

## Cadena causal H1 → H4 (evidencia LMB)

| Eslabón | No Alt. | Extreme Alt. | Δ | Hipótesis |
|---------|---------|--------------|---|-----------|
| InducedVertBreak (in) | 9.01 | 6.87 | −2.14 | H1 |
| Four-Seam IVB (in) | 17.1 | 13.1 | −4.0 | H1 |
| Whiff rate (swings) | 0.248 | 0.231 | −1.7pp | H1/H2 |
| Four-Seam whiff | 0.214 | 0.181 | **−3.3pp** | H1 (la que más depende de Magnus) |
| Changeup whiff | 0.292 | 0.279 | −1.3pp | H2 (depende de velo, no Magnus) |
| Chase rate | 0.481 | 0.465 | −1.6pp | — |
| Called strike | 0.298 | 0.282 | −1.6pp | — |
| Hard contact (EV≥95) | 0.120 | 0.150 | **+3.0pp** | H3 |
| HR rate | 0.0183 | 0.0227 | **+24%** | H3 |
| Exit velocity (mph) | 79.2 | 80.4 | +1.2 | H3 |
| Groundball rate | 0.267 | 0.308 | +4.1pp | H3 (ajuste) |
| Spin rate (rpm) | 2255 | 2292 | +37 (¡más spin, menos whiff!) | H4 |

## Interpretación

- **H1 (menos movimiento)**: confirmada. El Four-Seam pierde 4" de IVB y 3.3pp de
  whiff — el pitch cuyo "rise" depende más del efecto Magnus es el más castigado.
- **H2 (diferencial de velocidad gana valor relativo)**: confirmada. El changeup y
  los breaking pitches pierden *menos* whiff (su engaño es por diferencial de
  velocidad, no por Magnus) → ganan Stuff+ relativo en CDMX.
- **H3 (groundball vale más)**: confirmada. Hard contact +3pp y HR +24% en altitud
  → suprimir elevados (inducir groundballs) vale más en Harp Helú.
- **H4 (shape over label)**: confirmada. El spin rate SUBE en altitud pero el whiff
  BAJA → el spin no se traduce en movimiento real; lo que importa es el movimiento
  observado, no las rpm.

## Implicación operativa para Diablos Rojos

En Harp Helú conviene priorizar: (1) perfiles que inducen groundballs, (2) arsenales
basados en diferencial de velocidad (changeup, splitter) y en break horizontal que
sobrevive la menor densidad, y (3) no perseguir spin rate per se. El Four-Seam puro
es el perfil que más pierde en altitud.
