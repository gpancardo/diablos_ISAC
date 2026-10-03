# VAMOSDIABLOS — Stuff+ Model (Harp Helú)

Sistema de evaluación pitch-a-pitch de calidad de pitcheo **calibrado por altitud**
para Diablos Rojos / LMB. Hackathon ISAC 2026, Ciudad de México.

**Principio rector: cero fuga (no leakage).** Los targets (outcomes) jamás entran al
conjunto de features; las features de arsenal se computan solo desde el fold de
entrenamiento; la validación es agrupada (pitcher / temporada / estadio).

## Estructura

```
VAMOSDIABLOS/
├── stuff_model_df.parquet          # 635k lanzamientos LMB 2024–2026 (dato oficial)
├── stuff_plus_data_dictionary.csv  # diccionario de datos (feature_role)
├── pyproject.toml                  # deps (uv)
├── src/
│   ├── eda.py                      # perfilado + auditoría de fuga
│   ├── targets.py                  # construcción de targets (pesos lineales, flags)
│   ├── features.py                 # feature engineering (arsenal fold-aware, altitud)
│   ├── train.py                    # trainer CLI (modelo × target × features × holdout)
│   ├── run_experiments.py          # matriz R1 en paralelo
│   ├── stuff_score.py              # OOF -> score Plus (soporta invert= para run value)
│   ├── build_final_scores.py       # une Stuff+ / Location+ / Pitching+ por pitch
│   └── (R2) altitude_delta.py / shap
├── plans/
│   ├── DOCUMENTO_TECNICO.md        # documento técnico 6–10 pp (entregable #5)
│   ├── EXPERIMENT_PLAN.md          # plan de experimentos
│   ├── EDA_REPORT.md               # EDA + auditoría de fuga
│   ├── LITERATURE_REVIEW.md        # literatura 2025–2026
│   ├── ALTITUDE_FINDINGS.md        # efecto altitud (movimiento→outcome→run value)
│   ├── STABILITY.md                # fiabilidad cross-season + shrinkage
│   ├── STUFF_PLUS_RESULTS.md       # Stuff+ (Whiff+) y delta de altitud
│   └── RECOMENDACIONES.md          # recomendaciones operativas
├── dashboard/                       # Streamlit (premium Diablos Rojos)
│   ├── app.py
│   ├── models/whiff_model.joblib + feature_names/cat_levels
│   └── data/ (whiff_norm, altitude_delta, pitch_defaults, rankings)
├── literature/                      # abstracts arXiv verificados
├── results/                         # JSON + OOF parquet por experimento
└── models/                          # artefactos entrenados (R2)
```

## Uso

```bash
uv venv .venv && uv pip install -e .          # o: uv pip install -r pyproject.toml
.venv/bin/python src/eda.py                   # EDA
.venv/bin/python src/train.py --target whiff --model xgb --features pitching+ --holdout pitcher --n-folds 10
.venv/bin/python src/run_experiments.py       # matriz R1 en paralelo
.venv/bin/python src/build_final_scores.py    # Stuff+ / Location+ / Pitching+ por pitch
```

## Decisiones clave

- **Target principal** `xrun_value` = pesos lineales por evento (no RE24: no hay
  estado de bases ni id de PA en los datos — limitación documentada).
- **Señal fuerte** en los targets de outcome (whiff AUC ≈ 0.75) — el Stuff+ final
  los combina y normaliza a escala 100.
- **Robustez fuera de muestra** vía pitcher-holdout con prior de liga para pitchers
  no vistos (cold-start).

*Código en inglés, documentación en español. Sin fuga, sin claims no verificados.*
