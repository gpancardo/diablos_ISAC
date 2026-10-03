# Recomendaciones operativas — Diablos Rojos

Qué perfiles de pitch ganan y pierden Stuff+ en el Harp Helú (2,240 m), con la
física correspondiente. Respaldado por el modelo (Whiff+ AUC 0.745) y el análisis
empírico de altitud.

## 1. Qué PIERDE en CDMX (y por qué)

| Perfil | Δ Whiff+ | Mecanismo físico |
|--------|----------|------------------|
| Four-Seam puro (recta de 4 costuras) | **−3.7** | su "rise" (IVB −4.0") depende del efecto Magnus, que la menor densidad del aire reduce ~20% |
| Sinker | −1.7 | menos dependiente del Magnus pero aún castigado |

→ El perfil más perjudicado es la **recta cuyo valor depende del carry/rise**.

## 2. Qué GANA valor relativo (o resiste)

| Perfil | Δ Whiff+ | Mecanismo físico |
|--------|----------|------------------|
| Slider | −0.7 | el break horizontal/vertical por Magnus es menor, pero su engaño depende menos del carry |
| Curveball | −0.8 | ídem |
| Changeup | −0.9 | engaña por **diferencial de velocidad**, independiente del Magnus (H2) |
| Cutter | −0.3 | ídem |

→ En CDMX conviene **rotar el arsenal hacia break horizontal y diferenciales de
velocidad**, no hacia rectas de carry.

## 3. La regla general (H4 "shape over label")

El spin rate SUBE en altitud (2292 vs 2255 rpm) pero el movimiento BAJA. Perseguir
spin rate es inútil si no se traduce en break real. Lo que importa es el **movimiento
observado**, no las rpm. Nuestro modelo confirma: InducedVertBreak y HorzBreak
importan más que SpinRate para predecir whiff.

## 3b. Matiz importante: Pitching+/Out+ cae parejo, no solo el Four-Seam

El Δ Stuff+ (física) de la sección 1 es diferencial por forma de pitch, pero
el Δ Pitching+/Out+ (run value completo, que ya absorbe el +24% de HR) cae
**entre −2.2 y −2.9 puntos para TODO el arsenal**, Four-Seam incluido o no.
Conclusión para el cuerpo técnico: rotar hacia slider/changeup mejora el
whiff relativo, pero no "protege" del costo de carreras de la altitud — eso
solo lo hace la estrategia de groundball (punto 4). Ver
`plans/DOCUMENTO_TECNICO.md` §8.5 para la tabla completa de las 3 capas.

## 4. Groundball como estrategia de supresión (H3)

Hard contact +3pp y HR +24% en altitud. Un pitcher que induce groundballs reduce el
costo del batazo elevado. En Harp Helú, **el perfil groundball (sinker, splitter
bajo en la zona) gana valor relativo** como estrategia de supresión de carreras.

## 5. Perfil físico ideal en CDMX

Combinación de características que maximiza Stuff+ bajo Harp Helú (SHAP):
- **Velocidad alta** (RelSpeed ↑ → más whiff).
- **Break horizontal alto** (HorzBreak ↑) — resiste mejor la menor densidad.
- **Release alto** (RelHeight ↑) y **approach plano** (VertApprAngle menos negativo).
- **Evitar depender del IVB** (es lo que la altitud mata).
- Complementar con **changeup/splitter** (diferencial de velocidad) y **slider** (break horizontal).

## 6. Perfilado de prospectos y agentes libres

Para Diablos Rojos, priorizar prospectos/agentes libres cuyo arsenal:
1. Se base en **break horizontal** y **diferencial de velocidad** (sobrevive a CDMX).
2. Tenga **velocidad alta** (no se degrada por altitud).
3. Induzca **groundballs** (mitiga el +24% de HR en altitud).
4. **No dependa del carry** de la recta de 4 costuras.
