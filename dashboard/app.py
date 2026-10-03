"""Stuff+ Simulador — Diablos Rojos del México (Harp Helú).

Premium Streamlit dashboard.  The pitching coach enters a pitch's physical
characteristics and gets the Stuff+ estimate at sea level vs Harp Helú, plus the
arsenal x altitude map, pitcher rankings, and the physics that drive whiff.

Run:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = os.path.dirname(__file__)

# --------------------------------------------------------------------------
# Diablos Rojos palette
# --------------------------------------------------------------------------
RED = "#E4002B"
RED_HOT = "#ff2e55"
GOLD = "#d4af37"
BG = "#0a0e14"
CARD = "#12171f"
BORDER = "#1f2833"
TEXT = "#e6edf3"
MUTED = "#8b949e"

st.set_page_config(page_title="Stuff+ Diablos Rojos", layout="wide",
                   page_icon="⚾", initial_sidebar_state="expanded")

CSS = f"""
<style>
  .stApp {{ background: {BG}; color: {TEXT}; }}
  [data-testid="stSidebar"] {{ background: {CARD}; border-right: 1px solid {BORDER}; }}
  h1,h2,h3 {{ color: {TEXT}; font-weight: 700; letter-spacing: -0.02em; }}
  .brand {{ color: {RED}; }}
  .metric-card {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 14px;
    padding: 22px 26px; box-shadow: 0 8px 24px rgba(0,0,0,0.45);
  }}
  .metric-label {{ color: {MUTED}; font-size: 0.78rem; text-transform: uppercase;
    letter-spacing: 0.08em; font-weight: 600; }}
  .metric-value {{ font-size: 2.6rem; font-weight: 800; line-height: 1.05; }}
  .metric-sub {{ color: {MUTED}; font-size: 0.82rem; margin-top: 4px; }}
  .tag {{ display:inline-block; padding:2px 10px; border-radius:999px;
    font-size:0.72rem; font-weight:700; letter-spacing:0.05em; }}
  .tag-red {{ background: rgba(228,0,43,0.16); color: {RED_HOT}; border:1px solid rgba(228,0,43,0.4); }}
  .tag-gold {{ background: rgba(212,175,55,0.14); color: {GOLD}; border:1px solid rgba(212,175,55,0.4); }}
  .tag-green {{ background: rgba(46,204,113,0.14); color:#2ecc71; border:1px solid rgba(46,204,113,0.4); }}
  .footer {{ color: {MUTED}; font-size:0.78rem; margin-top:30px; border-top:1px solid {BORDER}; padding-top:14px; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE, "models", "whiff_model.joblib"))


@st.cache_data
def load_assets():
    names = json.load(open(os.path.join(BASE, "models", "feature_names.json")))
    cat_levels = json.load(open(os.path.join(BASE, "models", "cat_levels.json")))
    norm = pd.read_csv(os.path.join(BASE, "data", "whiff_norm.csv"))
    alt_delta = pd.read_csv(os.path.join(BASE, "data", "altitude_delta.csv"))
    defaults = pd.read_csv(os.path.join(BASE, "data", "pitch_defaults.csv"), index_col=0)
    pitchers = pd.read_csv(os.path.join(BASE, "data", "whiff_plus_by_pitcher.csv"))
    arsenal = pd.read_csv(os.path.join(BASE, "data", "whiff_plus_by_arsenal.csv"))
    shap_imp = pd.read_csv(os.path.join(BASE, "data", "shap_whiff_importance.csv"), index_col=0)
    # three-layer deliverable (Stuff+ / Location+ / Pitching+), optional: only
    # present once src/build_final_scores.py has been run.
    layer_delta_path = os.path.join(BASE, "data", "layer_altitude_delta.csv")
    layer_delta = pd.read_csv(layer_delta_path) if os.path.exists(layer_delta_path) else None
    final_pitcher_path = os.path.join(BASE, "data", "final_scores_by_pitcher.csv")
    final_pitchers = pd.read_csv(final_pitcher_path) if os.path.exists(final_pitcher_path) else None
    return (names, cat_levels, norm, alt_delta, defaults, pitchers, arsenal, shap_imp,
            layer_delta, final_pitchers)


NUM_FEATURES = ["RelSpeed", "EffectiveVelo", "SpinRate", "SpinAxis", "RelHeight",
                "RelSide", "Extension", "InducedVertBreak", "HorzBreak", "VertBreak",
                "VertRelAngle", "HorzRelAngle", "VertApprAngle", "HorzApprAngle",
                "SpeedDrop", "pfxx", "pfxz"]

# features the coach controls (the rest default to league-average for the pitch type)
CONTROL_FEATURES = ["RelSpeed", "SpinRate", "SpinAxis", "RelHeight", "Extension",
                    "InducedVertBreak", "HorzBreak"]


def build_input(pitch_type, pitcher_hand, batter_side, controls, names, cat_levels, defaults):
    row = {c: float(defaults.loc[pitch_type, c]) for c in NUM_FEATURES}
    for k, v in controls.items():
        row[k] = v
    row["EffectiveVelo"] = controls["RelSpeed"]  # keep consistent with velocity
    # one-hot categoricals
    cats = {}
    for c in ["AutoPitchType", "PitcherThrows", "BatterSide"]:
        val = {"AutoPitchType": pitch_type, "PitcherThrows": pitcher_hand,
               "BatterSide": batter_side}[c]
        for lvl in cat_levels.get(c, []):
            cats[f"{c}_{lvl}"] = 1.0 if lvl == val else 0.0
    X = pd.DataFrame([{**row, **cats}])[names]
    return X


def whiff_plus(p_whiff, pitch_type, norm):
    r = norm[norm.pitch_type == pitch_type]
    if r.empty:
        return 100.0
    mean, std = r.iloc[0]["mean"], r.iloc[0]["std"]
    if pd.isna(std) or std <= 0:
        return 100.0
    return 100 + 10 * (p_whiff - mean) / std


(names, cat_levels, norm, alt_delta, defaults, pitchers, arsenal, shap_imp,
 layer_delta, final_pitchers) = load_assets()
model = load_model()
# keep only pitch types with enough support (drop 1-pitch curiosities: Other,
# OneSeamFastBall, Knuckleball, Sweeper, TwoSeamFastBall)
PITCH_TYPES = [p for p in defaults.index
               if p in set(norm.pitch_type) and p in set(alt_delta.pitch_type)
               and not alt_delta.loc[alt_delta.pitch_type == p, "delta"].isna().all()]

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    f"<div style='display:flex;align-items:baseline;gap:14px;'>"
    f"<span style='font-size:1.6rem;font-weight:800;color:{RED};'>DIABLOS ROJOS</span>"
    f"<span style='font-size:1.6rem;font-weight:800;color:{TEXT};'>· Stuff+ Simulador</span>"
    f"</div>", unsafe_allow_html=True)
st.markdown(
    f"<div style='color:{MUTED};margin-bottom:22px;'>Calidad de pitcheo ajustada por "
    f"altitud — Harp Helú (2,240 m) vs nivel del mar · Liga Mexicana de Béisbol · "
    f"Hackathon ISAC 2026</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Sidebar — simulator controls
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"<span class='brand' style='font-weight:800;'>SIMULADOR DE PITCH</span>",
                unsafe_allow_html=True)
    st.caption("Ingresa las características físicas del lanzamiento.")

    pitch_type = st.selectbox("Tipo de pitch", PITCH_TYPES, index=3)
    c1, c2 = st.columns(2)
    pitcher_hand = c1.selectbox("Mano pitcher", ["Right", "Left"])
    batter_side = c2.selectbox("Lado bateador", ["Right", "Left", "Switch"])

    st.markdown("---")
    d = defaults.loc[pitch_type]
    controls = {
        "RelSpeed": st.slider("Velocidad (mph)", 70.0, 102.0, float(d["RelSpeed"]), 0.5),
        "SpinRate": st.slider("Spin rate (rpm)", 1200.0, 3200.0, float(d["SpinRate"]), 25.0),
        "InducedVertBreak": st.slider("IVB / rise (in)", -15.0, 30.0, float(d["InducedVertBreak"]), 0.5),
        "HorzBreak": st.slider("Break horizontal (in)", -20.0, 20.0, float(d["HorzBreak"]), 0.5),
        "RelHeight": st.slider("Altura de release (ft)", 4.0, 7.5, float(d["RelHeight"]), 0.05),
        "Extension": st.slider("Extensión (ft)", 4.0, 8.0, float(d["Extension"]), 0.05),
        "SpinAxis": st.slider("Eje de spin (°)", 0.0, 360.0, float(d["SpinAxis"]), 5.0),
    }

X = build_input(pitch_type, pitcher_hand, batter_side, controls, names, cat_levels, defaults)
p_whiff = float(model.predict_proba(X)[:, 1][0])
wp_sea = whiff_plus(p_whiff, pitch_type, norm)
delta = alt_delta.loc[alt_delta.pitch_type == pitch_type, "delta"]
delta = float(delta.iloc[0]) if not delta.empty else 0.0
delta = 0.0 if pd.isna(delta) else delta
wp_alt = wp_sea + delta

# --------------------------------------------------------------------------
# Simulator result
# --------------------------------------------------------------------------
st.markdown("### Resultado del simulador")
c1, c2, c3 = st.columns(3)

def grade(sp):
    if sp >= 110: return "Elite", RED_HOT, "tag-red"
    if sp >= 105: return "Plus", GOLD, "tag-gold"
    if sp >= 100: return "Sobre promedio", "#2ecc71", "tag-green"
    if sp >= 95: return "Promedio", MUTED, ""
    return "Bajo promedio", MUTED, ""

label, color, _ = grade(wp_sea)
with c1:
    st.markdown(f"""<div class='metric-card'>
      <div class='metric-label'>Stuff+ nivel del mar</div>
      <div class='metric-value' style='color:{color};'>{wp_sea:.1f}</div>
      <div class='metric-sub'>P(whiff) = {p_whiff:.1%} · {label}</div>
    </div>""", unsafe_allow_html=True)

label, color, _ = grade(wp_alt)
with c2:
    st.markdown(f"""<div class='metric-card'>
      <div class='metric-label'>Stuff+ Harp Helú</div>
      <div class='metric-value' style='color:{color};'>{wp_alt:.1f}</div>
      <div class='metric-sub'>{'pierde ' if delta<0 else 'gana '}{abs(delta):.1f} pts por altitud</div>
    </div>""", unsafe_allow_html=True)

with c3:
    dcol = RED if delta < -1 else GOLD if delta < -0.3 else "#2ecc71"
    st.markdown(f"""<div class='metric-card'>
      <div class='metric-label'>Δ altitud (Harp Helú − mar)</div>
      <div class='metric-value' style='color:{dcol};'>{delta:+.1f}</div>
      <div class='metric-sub'>pts de Stuff+ · {pitch_type}</div>
    </div>""", unsafe_allow_html=True)

st.caption("Escala: 100 = promedio LMB por tipo de pitch. 110 = +1 desviación. "
           "El delta de altitud es empírico (whiff por tipo de pitch).")

# --------------------------------------------------------------------------
# Movement map
# --------------------------------------------------------------------------
st.markdown("### Mapa de movimiento (efectividad por shape)")
col_l, col_r = st.columns([2, 1])

with col_l:
    # scan movement space for this pitch type (velocity/spin at defaults)
    ivb_span = np.linspace(d["InducedVertBreak"] - 12, d["InducedVertBreak"] + 12, 16)
    hb_span = np.linspace(d["HorzBreak"] - 12, d["HorzBreak"] + 12, 16)
    grid = []
    for ivb in ivb_span:
        for hb in hb_span:
            c2c = dict(controls, InducedVertBreak=ivb, HorzBreak=hb)
            Xg = build_input(pitch_type, pitcher_hand, batter_side, c2c, names, cat_levels, defaults)
            pg = float(model.predict_proba(Xg)[:, 1][0])
            grid.append((ivb, hb, whiff_plus(pg, pitch_type, norm)))
    gdf = pd.DataFrame(grid, columns=["IVB", "HB", "Whiff+"])
    fig = px.density_heatmap(gdf, x="HB", y="IVB", z="Whiff+", nbinsx=16, nbinsy=16,
                             color_continuous_scale="RdBu_r", range_color=[95, 115],
                             labels={"Whiff+": "Whiff+"})
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, font_color=TEXT,
                      height=420, margin=dict(l=10, r=10, t=10, b=10),
                      coloraxis_colorbar=dict(title="Whiff+"))
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    st.markdown("#### Perfil físico ideal")
    top = shap_imp.iloc[:12]
    top = top.sort_values(top.columns[0])
    fig2 = go.Figure(go.Bar(x=top[top.columns[0]], y=top.index, orientation="h",
                            marker_color=RED))
    fig2.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, font_color=TEXT, height=420,
                       margin=dict(l=10, r=10, t=10, b=10), xaxis_title="importancia |SHAP|")
    st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------------------------------
# Arsenal x altitude
# --------------------------------------------------------------------------
st.markdown("### Arsenal × Altitud — ¿qué gana y qué pierde en CDMX?")
piv = alt_delta.set_index("pitch_type").sort_values("delta")
fig3 = go.Figure()
fig3.add_trace(go.Bar(y=piv.index, x=piv["delta"], orientation="h",
                      marker_color=[RED if x < -1 else GOLD if x < -0.3 else "#2ecc71"
                                    for x in piv["delta"]]))
fig3.add_vline(x=0, line_color=MUTED, line_width=1)
fig3.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, font_color=TEXT, height=380,
                   margin=dict(l=10, r=10, t=10, b=10),
                   xaxis_title="Δ Whiff+ (Harp Helú − nivel del mar)",
                   title="El Four-Seam pierde más; slider/curva/changeup resisten")
st.plotly_chart(fig3, use_container_width=True)

# --------------------------------------------------------------------------
# Stuff+ vs Location+ vs Pitching+ -- the brief's three sub-models, side by side
# --------------------------------------------------------------------------
if layer_delta is not None:
    st.markdown("### Stuff+ vs Location+ vs Pitching+ — ¿dónde pega la altitud?")
    st.caption("Los tres sub-modelos del marco FanGraphs. Stuff+ (física pura, "
               "escala whiff) y Location+/Pitching+ (escala run value, invertida: "
               "100+ = menos carreras permitidas) se normalizan cada uno dentro de "
               "su propia escala por tipo de pitch × temporada.")
    LAYER_LABEL = {"stuff_plus": "Stuff+ (física)", "location_plus": "Location+ (ubicación)",
                   "pitching_plus": "Pitching+/Out+ (combinado)"}
    LAYER_COLOR = {"stuff_plus": RED, "location_plus": GOLD, "pitching_plus": "#2ecc71"}
    fig3b = go.Figure()
    for layer in ["stuff_plus", "location_plus", "pitching_plus"]:
        d = layer_delta[layer_delta.layer == layer].sort_values("pitch_type")
        fig3b.add_trace(go.Bar(name=LAYER_LABEL[layer], x=d["pitch_type"], y=d["delta"],
                               marker_color=LAYER_COLOR[layer]))
    fig3b.add_hline(y=0, line_color=MUTED, line_width=1)
    fig3b.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, font_color=TEXT, height=400,
                        margin=dict(l=10, r=10, t=10, b=10), barmode="group",
                        yaxis_title="Δ Plus score (Harp Helú − nivel del mar)",
                        legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig3b, use_container_width=True)

# --------------------------------------------------------------------------
# Pitcher rankings
# --------------------------------------------------------------------------
st.markdown("### Ranking de pitchers por Whiff+")
top_p = pitchers[pitchers.n_pitches >= 200].sort_values("stuff_plus", ascending=False).head(20)
fig4 = go.Figure(go.Bar(x=top_p["stuff_plus"], y=top_p["pitcher"], orientation="h",
                        marker_color=[RED if s >= 106 else GOLD if s >= 103 else MUTED
                                      for s in top_p["stuff_plus"]]))
fig4.add_vline(x=100, line_color=MUTED, line_width=1)
fig4.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, font_color=TEXT, height=460,
                   margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Whiff+",
                   yaxis=dict(autorange="reversed"))
st.plotly_chart(fig4, use_container_width=True)

if final_pitchers is not None:
    st.markdown("### Ranking de pitchers por Pitching+/Out+ (combinado)")
    st.caption("Run value permitido, invertido a escala plus -- el número que de "
               "verdad responde \"¿quién previene más carreras?\".")
    top_pp = final_pitchers[final_pitchers.n_pitches >= 200].sort_values(
        "pitching_plus", ascending=False).head(20)
    fig4b = go.Figure(go.Bar(x=top_pp["pitching_plus"], y=top_pp["pitcher"], orientation="h",
                             marker_color=[RED if s >= 106 else GOLD if s >= 103 else MUTED
                                           for s in top_pp["pitching_plus"]]))
    fig4b.add_vline(x=100, line_color=MUTED, line_width=1)
    fig4b.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, font_color=TEXT, height=460,
                        margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Pitching+",
                        yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig4b, use_container_width=True)

st.markdown(f"<div class='footer'>Modelo: XGBoost sobre física de release · validación "
            f"pitcher-holdout (10-fold) sin fuga · whiff AUC 0.745, ECE 0.007 · "
            f"Diablos Rojos del México — análisis de rendimiento deportivo, uso interno.</div>",
            unsafe_allow_html=True)
