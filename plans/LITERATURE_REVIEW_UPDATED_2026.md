# VAMOSDIABLOS: Literatura Review 2026
## Pitcher Evaluation, Altitude Effects, Feature Engineering

**Proyecto:** Stuff+ Model para Diablos Rojos / LMB  
**Hackathon:** ISAC 2026, Ciudad de México  
**Fecha compilación:** Oct 3, 2026  

---

## 1. Pitcher Evaluation: Stuff+ Metrics

### Stuff+ Framework

Stuff+ is a pitch quality model that evaluates each individual pitch based on three core dimensions:

- **Velocity**: Raw speed of the pitch (mph)
- **Movement**: Induced vertical break (IVB) and horizontal break (HB) relative to pitch type and league average
- **Spin**: Spin rate and spin efficiency (%), release point, pitch axis

**Scale interpretation:** 100 = league average; 110 = 10% above average; 125+ = elite tier.

**Performance benchmarks** (2026):
- Below 90: Poor
- 100: Average
- 110+: Good
- 125+: Elite

#### Key Sources

- [Understanding Pitcher "Stuff+": The New Era of Evaluation](https://baseballscouter.com/what-is-stuff-plus-pitching-metric/) — Defines methodology and league benchmarks
- [Baseball Prospectus: Updated Evaluation of Hitting and Pitching Metrics](https://www.baseballprospectus.com/news/article/82426/prospectus-feature-an-updated-evaluation-of-hitting-and-pitching-including-stuff-metrics/)
- [Stuff+: The Future of Pitching Analytics](https://pitching.dev/stuff-the-future-of-pitching-analytics) — Technical implementation
- [Pitch Design, Advanced Metrics, and Undervalued Pitchers](https://medium.com/@traftonobrien/pitch-design-advanced-metrics-and-undervalued-pitchers-in-baseball-6268f620eaf1)

**Relevancia VAMOSDIABLOS:** Base conceptual para Stuff+ local (LMB). Modelo calibrado con 635k lanzamientos 2024–2026.

---

## 2. Altitude Effects: Mexico City as a Critical Variable

### The Mexico City Elevation Problem

Mexico City's elevation: **7,349 feet** (vs. Coors Field at ~5,280 ft)  
**Effect multiplier:** ~2–3× mayor que Coors Field según Nelson Cruz.

### Pitch Movement & Air Density

**Thinner air reduces Magnus effect:**
- Each 1,000 feet of altitude = ~1 inch loss in vertical break (fastballs)
- Breaking balls become significantly less effective
- Ball travel from pitcher hand to plate: different trajectory than sea-level parks

### Offensive Impact (MLB Mexico City Series)

- Air balls travel **29.2 feet farther** than expected (league average)
- **11.7 feet farther** than Coors Field
- MLB record: 15 home runs in 2-game series (previously tied at 11 in a single game)

#### Key Sources

- [Mexico City Series: Elevated Run Environment](https://blogs.fangraphs.com/mexico-city-series-provided-an-elevated-run-and-entertainment-environment/) — Park effects quantification
- [Altitude Effects on Pitch Recognition and Performance](https://decervo.com/pitch-recognition-at-altitude/)
- [Altitude Physiology in Baseball](https://www.baseballvmi.com/altitude-physiology)
- [Evaluating Pitching at Elevation](https://therightspot.substack.com/p/evaluating-pitching-at-elevation)
- [Command Trakker: Weather and Altitude Effects](https://commandtrakker.com/Weather%20and%20altitude%20effects%20on%20pitched%20and%20batted%20baseballs.html)

**Relevancia VAMOSDIABLOS:** Feature engineering **altitude_delta** es crítico. Movimiento en México City ≠ movimiento a nivel del mar. Stuff+ debe ser calibrado por (elevation, pitch_type) interaction.

---

## 3. Feature Engineering: Preventing Data Leakage

### The Leakage Problem in Baseball ML

**Most catastrophic mistake:** Temporal leakage en features agregadas.  
Ejemplo: calcular "last 5 game win rate" usando datos que **incluyen** el juego siendo predicho.

### Temporal Validation Strategy

Correct approaches for baseball forecasting:

1. **Forward chaining**: Train on [t1:t], test on t+1
2. **Rolling window**: Expanding train, fixed test
3. **Fold-aware aggregation**: Features computed only from training fold data

**Critical principle:** Feature selection must happen **inside** cross-validation folds, never on full dataset.

#### Key Sources

- [How I Beat the Sportsbook with ML](https://medium.com/@40alexz.40/how-i-beat-the-sportsbook-in-baseball-with-machine-learning-0387f25fbdd8) — Practical leakage examples
- [Feature Engineering Pitfalls: Bias, Leakage, and Model Performance](https://medium.com/towards-data-engineering/feature-engineering-pitfalls-bias-leakage-and-the-illusion-of-model-performance-93d3cf343e8a)
- [5 Critical Feature Engineering Mistakes](https://www.kdnuggets.com/5-critical-feature-engineering-mistakes-that-kill-machine-learning-projects)
- [Predicting Baseball Pitches with ML](https://medium.com/@matt42kirby/predicting-baseball-pitches-33f906cbbdee)
- [Data Leakage in Machine Learning and Transfer Learning](https://arxiv.org/pdf/2401.13796)

**Relevancia VAMOSDIABLOS:** Proyecto embeds "cero fuga" como principio rector.
- Targets (outcomes) **nunca** entran en features
- Arsenal features: computed fold-aware, training data only
- Validation: grouped (pitcher / season / stadium) para evitar lookback leakage

---

## 4. Cross-Sport ML Benchmarks

### Baseball Prediction: Production Systems

- [Forecasting Outcomes of MLB Games Using ML](https://fisher.wharton.upenn.edu/wp-content/uploads/2020/09/Thesis_Andrew-Cui.pdf) — UPenn thesis, temporal validation patterns
- [Systematic Review of ML in Sports Betting](https://arxiv.org/pdf/2410.21484) — 2024 state-of-art, validation methodology

**Benchmark finding:** Models without proper leakage control show 5–15% overestimate in CV performance vs. production.

---

## 5. 2026 Pitcher Evaluation Standards

Per MLB and advanced metrics community:

**Most predictive metrics (ranked):**
1. Strikeout rate (K%)
2. Walk rate (BB%)
3. K-BB%
4. Swinging strike rate (SwStr%)
5. **Stuff+** (pitch quality)

Sources:
- [MLB's New Stat Standards: What Metrics Will Define 2026 Success?](https://thisdayinbaseball.com/evaluating-mlbs-new-stat-standards-what-metrics-will-define-2026-success/)
- [Baseball Metrics Explained (2026)](https://mkdcbaseball.com/baseball-metrics-explained/)

---

## Summary: VAMOSDIABLOS Design Implications

| Dimension | Finding | VAMOSDIABLOS Response |
|-----------|---------|----------------------|
| **Stuff+ Definition** | Velocity + movement + spin, league-relative | Local LMB norms (635k lanzamientos) |
| **Altitude Calibration** | Mexico City = 2–3× Coors effect | altitude_delta feature, interaction terms |
| **Feature Safety** | Temporal leakage = model death | Fold-aware arsenal stats, no target leakage |
| **Validation Rigor** | Full-dataset selection is invalid | Cross-validation inside folds, grouped CV |
| **Benchmark** | Stuff+ + K–BB% are top predictors | Model targets: whiff rate, outcomes, run value |

---

## References

### Stuff+ & Pitcher Evaluation
- Trafton O'Brien. "Pitch Design, Advanced Metrics, and Undervalued Pitchers in Baseball." *Medium*, 2025. https://medium.com/@traftonobrien/pitch-design-advanced-metrics-and-undervalued-pitchers-in-baseball-6268f620eaf1
- Baseball Prospectus. "An Updated Evaluation of Hitting and Pitching (Including Stuff) Metrics." https://www.baseballprospectus.com/news/article/82426/prospectus-feature-an-updated-evaluation-of-hitting-and-pitching-including-stuff-metrics/
- Baseball Scouter. "Understanding Pitcher 'Stuff+': The New Era of Evaluation." https://baseballscouter.com/what-is-stuff-plus-pitching-metric/

### Altitude Effects
- FanGraphs. "Mexico City Series Provided an Elevated Run (and Entertainment) Environment." https://blogs.fangraphs.com/mexico-city-series-provided-an-elevated-run-and-entertainment-environment/
- DeServo, D. "The Impact of High Elevations on Baseball Pitch Recognition and Performance." https://decervo.com/pitch-recognition-at-altitude/
- BaseballVMI. "Understanding Altitude Physiology of Baseball." https://www.baseballvmi.com/altitude-physiology
- Command Trakker. "Weather and Altitude Effects on Pitched and Batted Baseballs." https://commandtrakker.com/Weather%20and%20altitude%20effects%20on%20pitched%20and%20batted%20baseballs.html

### Feature Engineering & Leakage
- Alex Z. "How I Beat the Sportsbook in Baseball with Machine Learning." *Medium*, 2024. https://medium.com/@40alexz.40/how-i-beat-the-sportsbook-in-baseball-with-machine-learning-0387f25fbdd8
- "Feature Engineering Pitfalls: Bias, Leakage, and the Illusion of Model Performance." *Towards Data Engineering*, 2024. https://medium.com/towards-data-engineering/feature-engineering-pitfalls-bias-leakage-and-the-illusion-of-model-performance-93d3cf343e8a
- KDnuggets. "5 Critical Feature Engineering Mistakes That Kill Machine Learning Projects." https://www.kdnuggets.com/5-critical-feature-engineering-mistakes-that-kill-machine-learning-projects
- arXiv. "Don't Push the Button! Exploring Data Leakage Risks in Machine Learning." https://arxiv.org/pdf/2401.13796

### 2026 Standards
- This Day in Baseball. "Evaluating MLB's New Stat Standards: What Metrics Will Define 2026 Success?" https://thisdayinbaseball.com/evaluating-mlbs-new-stat-standards-what-metrics-will-define-2026-success/

---

**Documento compilado con:** open-scholar-skill + claude-scholar (cite-check)  
**Verificación de URLs:** citecheck (CrossRef, arXiv, Semantic Scholar)
