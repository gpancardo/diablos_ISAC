# Revisión de Literatura (2025–2026) — Calidad de Pitcheo y Altitud

Estado del arte relevante al reto Stuff+ ajustado por entorno. **Todos los IDs de
arXiv y DOIs listados abajo fueron verificados contra arXiv / Crossref el
27-sep-2026.** Ninguna cita se da por buena sin fuente localizable.

## 1. Metodología Stuff+/Pitching+ (sabermetría moderna)

- **FanGraphs Stuff+/Location+/Pitching+** (Eno Sarris, Max Bay; 2022–2025).
  Marco canónico: modelo supervisado que estima el *run value esperado* de un
  pitch desde sus características físicas (velo, IVB, HB, spin, release),
  normalizado a escala 100 **relativo al tipo de pitch y a la liga**. La
  separación Stuff+/Location+/Pitching+ aísla calidad física, comando y
  contexto — el mismo desglose de tres capas que pide el reto.
- **Tango, Lichtman & Dolphin (2007), *The Book*.** Fundamento de count/base-out
  state y run expectancy (RE24) que subyace al run value.

## 2. Física de la altitud (la variable anchor)

- **SABR, "High Altitude Offense"** (*Baseball Research Journal*). Los estadios
  de gran altitud producen más carreras: menor densidad → menor resistencia y
  menor efecto Magnus → la bola vuela más y los breaking pitches se mueven
  menos. *Caveat: BRJ no registra DOIs; no verificable vía Crossref. Verificar
  en el sitio de SABR antes de citar en un documento formal.*
- **Alan Nathan.** "The effect of spin on the flight of a baseball",
  *American Journal of Physics* 76(2), 119–124 (2008), DOI 10.1119/1.2805242.
  Base de la física del Magnus. La cifra "un pitch de 18" cae a ~14–15" en
  Coors" que circula en este repo es una **estimación de densidad ISA** (18" ×
  ρ(Coors)/ρ₀ ≈ 14.5"), no una cita literal de Nathan; la dejamos explícita
  como cálculo propio (ver `src/` y `plans/METHODOLOGY.md`).
- **Densidad a 2,240 m.** Por la atmósfera estándar ISA, la presión a 2,240 m es
  ~0.76·p₀; con temperatura de CDMX (~22 °C) la densidad es **~74 % de la del
  nivel del mar (−26 %), no "−20 %"**. A Coors (1,609 m) es ~80 % (−20 %). Esta
  corrección es material para la calibración del reto.

## 3. Trabajos académicos (arXiv, verificados 27-sep-2026)

| Paper (título real) | arXiv | Aporte al reto |
|---|---|---|
| Neural Sabermetrics with World Model: Play-by-play Predictive Modeling with Large Language Model — Ahn, Young Jin et al. (2026) | 2602.07030 | Modelo generativo play-by-play vía LLM; secuencia como auto-regresión. Referencia "Neural sabermetrics" citada por el certamen. |
| Cross-individual generalizability of machine learning models for ball speed prediction in baseball pitching — Takamido, Ryota et al. (2026) | 2605.05487 | Evaluación leave-one-subject-out de generalización entre pitchers; alineada con el 25 % de robustez fuera de muestra. |
| Counterfactual Optimization of Baseball Pitch Sequences and Estimation of Its Impact on Season-Level Statistics — Takamido, Ryota et al. (2026) | 2606.17345 | Transformer + contrafactuales de secuencia. Respalda la capa de secuencia. |
| Structure of Pitch-Pattern Motifs in Major League Baseball — Park, Youngjai et al. (2026) | 2601.11904 | Motivos de secuencia de pitch; justifica features de secuencia/arsenal. |
| Tractable Algorithms for Changepoint Detection in Player Performance Metrics — Glazer, Amanda et al. (2025) | 2510.25961 | Detección de cambio en métricas; útil para consistencia pitcher. |
| A Hierarchical Bayesian Model of Pitch Framing — Deshpande & Wyner (2017) | 1704.00823 | Fundamento del Bayesiano jerárquico (shrinkage por pitcher). |
| openWAR: An Open Source System for Evaluating Overall Player Performance in Major League Baseball — Baumer, Jensen & Matthews (2013/2015) | 1312.7158 | Run value / RE24 open-source. |

## 4. Referencia citada por el certamen — verificado

- **McBride, J. (2026), "Measuring Pitcher Production Fairly in Baseball Using
  the Shapley Value"**, *Games* 17(2), art. 15, **DOI 10.3390/g17020015**.
  *(Nota: el título que circulaba en el repo — "Shapley value decomposition of
  pitching performance" — era incorrecto; el DOI de MDPI redirige por diseño, no
  es un 302 anómalo.)*

## 5. Correcciones a citas previas

- ~~"Huang & Hsu (2021), Big Data Analytics in Baseball: A Review, SAGE Open"~~ →
  **Huang, J.-H. & Hsu, Y.-C. (2021), "A Multidisciplinary Perspective on
  Publicly Available Sports Data in the Era of Big Data: A Scoping Review",
  *SAGE Open* 11(4), DOI 10.1177/21582440211061566.** El título anterior era
  inventado y la cita se usaba para sostener "shape over label", algo que esa
  review (sobre disponibilidad de datos deportivos) no establece. "Shape over
  label" se sostiene en la física (Nathan 2008) y en el EDA propio, no en esta
  cita.

## 6. Gap que cubre este proyecto

Los modelos Stuff+/Pitching+ de MLB fueron entrenados en estadios a nivel del
mar. **No existe** un equivalente calibrado para las condiciones físicas de la
LMB (densidad −26 %, Magnus reducido). Construirlo —cuantificando el delta
Stuff+ por pitch shape entre Harp Helú y nivel del mar— es la contribución
original del equipo.
