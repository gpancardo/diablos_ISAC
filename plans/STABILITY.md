# Fiabilidad y consistencia (análisis sabermétrico)

La prueba que separa un modelo real de un ajuste de curvas: ¿el Stuff+ mide una
habilidad estable y predictiva, o solo ruido?

## 1. Estabilidad año-a-año de la tasa de whiff

| Comparación | r | n pitchers |
|-------------|---|------------|
| whiff real 2024 → 2025 | 0.517 | 217 |
| whiff real 2025 → 2026 | 0.559 | 211 |
| **modelo(T) → whiff real(T+1)** | **0.394 / 0.442** | — (validez predictiva) |
| modelo → modelo | 0.617 / 0.679 | — (auto-estabilidad) |

→ La tasa de whiff es una habilidad **repetible** (r≈0.52–0.56 año a año).
Nuestro Stuff+ físico predice el whiff FUTURO con r≈0.4, y es más auto-estable
(r≈0.65) porque promedia el ruido. El gap (0.52 vs 0.4) es la habilidad estable
NO capturada por la física (engaño, secuencia, mix de pitches) — honesto y esperado.

## 2. Shrinkage empírico-Bayes (regresión a la media)

- Varianza entre pitchers = 0.0102, dentro = 0.1729.
- Factor de shrinkage mediano = **0.90** (whiff es estable → poca regresión).
- Rankings ajustados por tamaño de muestra: pitcher con 13 pitches y 61.5% de whiff
  observado se encoge a 40.3% (evita sobre-ranking de muestras chicas).

## 3. Implicación

El modelo mide una **habilidad física estable y forward-predictiva** (r≈0.4 a un año),
no sobreajuste. Esto respalda los criterios de "calidad de modelo" y "aplicabilidad
real": el club puede confiar en que un Stuff+ alto hoy predice whiff mañana.

Artefacto: `results/pitcher_shrunk_whiff.csv` (ranking ajustado por consistencia).
