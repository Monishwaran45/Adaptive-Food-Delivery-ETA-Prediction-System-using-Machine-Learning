"""
Enterprise-Grade Luxury Dark Glassmorphism Design System & UI Tokens
Adaptive Food Delivery ETA Intelligence System
"""

CUSTOM_CSS = """
<style>
/* Import Premium Modern Typography */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --bg-main: #07090e;
    --bg-card: rgba(15, 23, 42, 0.65);
    --bg-card-hover: rgba(30, 41, 59, 0.85);
    --border-card: rgba(255, 255, 255, 0.08);
    --border-glow: rgba(56, 189, 248, 0.25);
    --primary: #38bdf8;
    --primary-gradient: linear-gradient(135deg, #0284c7 0%, #2563eb 50%, #7c3aed 100%);
    --accent-indigo: #818cf8;
    --accent-emerald: #10b981;
    --accent-amber: #f59e0b;
    --accent-rose: #f43f5e;
    --font-heading: 'Outfit', sans-serif;
    --font-body: 'Plus Jakarta Sans', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
}

html, body, [class*="css"] {
    font-family: var(--font-body) !important;
}

/* Background Atmosphere */
.stApp {
    background-color: var(--bg-main) !important;
    background-image: 
        radial-gradient(at 0% 0%, rgba(14, 165, 233, 0.12) 0px, transparent 50%),
        radial-gradient(at 100% 0%, rgba(139, 92, 246, 0.12) 0px, transparent 50%),
        radial-gradient(at 50% 50%, rgba(15, 23, 42, 0.5) 0px, transparent 100%),
        radial-gradient(at 0% 100%, rgba(16, 185, 129, 0.06) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(244, 63, 94, 0.06) 0px, transparent 50%);
    background-attachment: fixed !important;
    color: #f8fafc !important;
}

/* Top Hero Command Center Header */
.hero-header {
    position: relative;
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.6) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 28px 36px;
    margin-bottom: 26px;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    overflow: hidden;
}

.hero-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc, #38bdf8);
    background-size: 200% auto;
    animation: gradientShimmer 4s linear infinite;
}

@keyframes gradientShimmer {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.hero-title {
    font-family: var(--font-heading) !important;
    font-weight: 800 !important;
    font-size: 2.35rem !important;
    letter-spacing: -0.03em !important;
    background: linear-gradient(135deg, #ffffff 20%, #93c5fd 60%, #c4b5fd 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
    line-height: 1.15;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 0.95rem;
    font-weight: 400;
    max-width: 800px;
    margin-bottom: 0;
}

/* Status Indicator Dot with Pulse */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.12);
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 600;
    color: #e2e8f0;
    backdrop-filter: blur(8px);
}

.pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #10b981;
    box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
    animation: pulse 1.8s infinite;
}

@keyframes pulse {
    0% {
        transform: scale(0.95);
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
    }
    70% {
        transform: scale(1);
        box-shadow: 0 0 0 8px rgba(16, 185, 129, 0);
    }
    100% {
        transform: scale(0.95);
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
    }
}

/* Pill Badges */
.badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(56, 189, 248, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.25);
    color: #38bdf8;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}

.badge-green {
    background: rgba(16, 185, 129, 0.12);
    border-color: rgba(16, 185, 129, 0.35);
    color: #34d399;
}

.badge-amber {
    background: rgba(245, 158, 11, 0.12);
    border-color: rgba(245, 158, 11, 0.35);
    color: #fbbf24;
}

.badge-red {
    background: rgba(239, 68, 68, 0.12);
    border-color: rgba(239, 68, 68, 0.35);
    color: #f87171;
}

/* Glassmorphic Glowing KPI Cards */
.kpi-card {
    position: relative;
    background: linear-gradient(145deg, rgba(20, 29, 47, 0.7) 0%, rgba(11, 18, 33, 0.85) 100%);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 18px;
    padding: 22px 24px;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    overflow: hidden;
    height: 100%;
}

.kpi-card:hover {
    transform: translateY(-4px);
    border-color: rgba(56, 189, 248, 0.35);
    box-shadow: 0 16px 36px -8px rgba(0, 0, 0, 0.6), 0 0 20px rgba(56, 189, 248, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.1);
}

.kpi-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}

.kpi-title {
    font-family: var(--font-heading);
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
}

.kpi-icon-wrap {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.05);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.kpi-value {
    font-family: var(--font-heading);
    font-size: 2.2rem;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -0.02em;
    line-height: 1.05;
    margin-bottom: 6px;
}

.kpi-unit {
    font-size: 1.05rem;
    font-weight: 500;
    color: #94a3b8;
    margin-left: 3px;
}

.kpi-footer {
    font-size: 0.82rem;
    color: #64748b;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Operational Dispatch Alert Card */
.dispatch-action-card {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.7) 100%);
    border-left: 5px solid #38bdf8;
    border-radius: 14px;
    padding: 18px 22px;
    margin: 20px 0;
    backdrop-filter: blur(12px);
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.35);
}

.dispatch-action-title {
    font-family: var(--font-heading);
    font-weight: 700;
    font-size: 1rem;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.dispatch-action-desc {
    color: #cbd5e1;
    font-size: 0.92rem;
    line-height: 1.5;
}

/* 4-Stage Delivery Journey Step Nodes */
.journey-container {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin: 20px 0;
}

.journey-node {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 16px;
    text-align: center;
    position: relative;
    transition: all 0.2s ease;
}

.journey-node:hover {
    border-color: rgba(56, 189, 248, 0.4);
    background: rgba(30, 41, 59, 0.7);
}

.journey-node-icon {
    font-size: 1.5rem;
    margin-bottom: 6px;
}

.journey-node-title {
    font-family: var(--font-heading);
    font-weight: 700;
    font-size: 0.9rem;
    color: #f1f5f9;
    margin-bottom: 2px;
}

.journey-node-pct {
    font-family: var(--font-mono);
    font-size: 1.1rem;
    font-weight: 700;
    color: #38bdf8;
}

.journey-node-desc {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 4px;
}

/* Tab Bar Refinement */
div[data-baseweb="tab-list"] {
    background: rgba(15, 23, 42, 0.75) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
    padding: 6px !important;
    gap: 8px !important;
    backdrop-filter: blur(12px) !important;
}

div[data-baseweb="tab"] {
    border-radius: 10px !important;
    font-family: var(--font-heading) !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    color: #94a3b8 !important;
    padding: 10px 22px !important;
    border: none !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

div[data-baseweb="tab"]:hover {
    color: #f8fafc !important;
    background: rgba(255, 255, 255, 0.04) !important;
}

div[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 18px rgba(37, 99, 235, 0.45) !important;
}

/* Streamlit Native Elements Polish */
.stButton > button {
    font-family: var(--font-heading) !important;
    background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    padding: 12px 24px !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 16px rgba(37, 99, 235, 0.4) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    width: 100% !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #0369a1 0%, #1d4ed8 100%) !important;
    box-shadow: 0 8px 24px rgba(37, 99, 235, 0.6) !important;
    transform: translateY(-2px) !important;
}

/* Sidebar Custom Styling */
section[data-testid="stSidebar"] {
    background: #080c14 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
}

section[data-testid="stSidebar"] div.stSelectbox, 
section[data-testid="stSidebar"] div.stSlider,
section[data-testid="stSidebar"] div.stNumberInput {
    margin-bottom: 8px;
}

/* Expander Styling */
.streamlit-expanderHeader {
    font-family: var(--font-heading) !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    background-color: rgba(15, 23, 42, 0.5) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
}

/* Dataframe & Tables */
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
    overflow: hidden !important;
}

/* Code Snippets */
code {
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 6px !important;
    color: #38bdf8 !important;
    font-family: var(--font-mono) !important;
    font-size: 0.88em !important;
}
</style>
"""


def clean_html(html_str: str) -> str:
    """Removes leading and trailing whitespace so Markdown never parses HTML as code blocks."""
    return "\n".join(line.strip() for line in html_str.strip().splitlines() if line.strip())


def render_hero_header() -> str:
    """Renders the command center top hero banner."""
    return (
        '<div class="hero-header">'
        '<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px;">'
        '<div>'
        '<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
        '<span class="status-pill"><span class="pulse-dot"></span>LIVE ML DISPATCH ENGINE v3.0</span>'
        '<span class="badge-pill badge-green">⚡ Real-Time Inference</span>'
        '</div>'
        '<div class="hero-title">Adaptive Food Delivery ETA Intelligence</div>'
        '<div class="hero-subtitle">4-Stage Operational Feature Decomposition • XGBoost Regression • 95% Conformal Prediction Intervals • Proactive Delay Risk Profiling • SHAP XAI</div>'
        '</div>'
        '<div style="display:flex;flex-direction:column;gap:8px;align-items:flex-end;">'
        '<div style="display:flex;gap:8px;flex-wrap:wrap;">'
        '<span class="badge-pill">XGBoost Engine</span>'
        '<span class="badge-pill badge-green">R² = 0.8336</span>'
        '<span class="badge-pill badge-amber">MAE = 3.05m</span>'
        '<span class="badge-pill">RMSE = 3.83m</span>'
        '<span class="badge-pill">Risk Acc = 86.2%</span>'
        '</div>'
        '<div style="color:#64748b;font-size:0.78rem;font-family:var(--font-mono);">Coverage: 95.0% Empirical Test Coverage (±7.66m) • 35 Cross-Stage Signals</div>'
        '</div>'
        '</div>'
        '</div>'
    )


def render_kpi_card(title: str, value: str, unit: str, subtitle: str, icon: str, color_hex: str = "#38bdf8") -> str:
    """Renders a modern glassmorphic KPI Card."""
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-header">'
        f'<span class="kpi-title">{title}</span>'
        f'<div class="kpi-icon-wrap" style="color:{color_hex};">{icon}</div>'
        f'</div>'
        f'<div class="kpi-value" style="color:{color_hex};">{value}<span class="kpi-unit">{unit}</span></div>'
        f'<div class="kpi-footer"><span>{subtitle}</span></div>'
        f'</div>'
    )


def render_risk_kpi_card(risk_result: dict) -> str:
    """Renders the Delay Risk KPI card with all 3 probability percentages."""
    probs = risk_result["probabilities"]
    risk_label = risk_result["risk_label"]
    risk_color = risk_result["risk_color"]
    
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-header">'
        f'<span class="kpi-title">Delay Risk Profile</span>'
        f'<div class="kpi-icon-wrap" style="color:{risk_color};">🚦</div>'
        f'</div>'
        f'<div class="kpi-value" style="color:{risk_color};font-size:1.8rem;">{risk_label.replace(" Risk", "")} <span style="font-size:1rem;font-weight:500;">Risk</span></div>'
        f'<div style="margin-top:6px;font-size:0.78rem;color:#cbd5e1;font-family:var(--font-mono);line-height:1.5;">'
        f'<div><span style="color:#10b981;">🟢 Low:</span> <b>{probs["Low Risk"]:.1f}%</b> &nbsp; <span style="color:#f59e0b;">🟠 Med:</span> <b>{probs["Medium Risk"]:.1f}%</b> &nbsp; <span style="color:#ef4444;">🔴 High:</span> <b>{probs["High Risk"]:.1f}%</b></div>'
        f'</div>'
        f'</div>'
    )


def render_dispatch_protocol_card(risk_result: dict, dist_km: float, traffic: str, weather: str, rating: float) -> str:
    """Renders the rich Decision-Intelligent Automated Logistics Protocol Recommendation card."""
    risk_label = risk_result["risk_label"]
    risk_color = risk_result["risk_color"]
    top_prob = max(risk_result["probabilities"].values())
    advice = risk_result["advice"]
    
    protocol_badge = "🟢 OPTIMAL DISPATCH PROTOCOL" if "Low" in risk_label else "🟠 BALANCED CONTINGENCY PROTOCOL" if "Medium" in risk_label else "🔴 CRITICAL DISPATCH OVERRIDE"

    return (
        f'<div class="dispatch-action-card" style="border-left-color:{risk_color};">'
        f'<div class="dispatch-action-title" style="color:{risk_color};margin-bottom:8px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">'
        f'<span>⚡ Automated Logistics Protocol Recommendation:</span>'
        f'<span style="font-size:0.82rem;font-weight:800;background:rgba(15,23,42,0.8);border:1px solid {risk_color}44;padding:3px 10px;border-radius:9999px;">{protocol_badge}</span>'
        f'</div>'
        f'<div style="font-size:0.83rem;color:#cbd5e1;margin-bottom:6px;font-family:var(--font-mono);line-height:1.6;background:rgba(0,0,0,0.25);padding:6px 12px;border-radius:8px;border:1px solid rgba(255,255,255,0.05);">'
        f'<span>● Risk: <b>{risk_label} ({top_prob:.1f}%)</b></span> &nbsp;•&nbsp; '
        f'<span>● Route: <b>{dist_km:.2f} km</b></span> &nbsp;•&nbsp; '
        f'<span>● Traffic: <b>{traffic}</b></span> &nbsp;•&nbsp; '
        f'<span>● Weather: <b>{weather}</b></span> &nbsp;•&nbsp; '
        f'<span>● Courier: <b>{rating:.1f}★ Rating</b></span>'
        f'</div>'
        f'<div class="dispatch-action-desc" style="margin-top:6px;font-size:0.88rem;color:#f1f5f9;">'
        f'{advice}'
        f'</div>'
        f'</div>'
    )


def render_journey_timeline(stage_contributions: list[dict]) -> str:
    """Renders a modern 4-step operational feature decomposition journey."""
    pct_map = {item["Stage"]: item["Contribution (%)"] for item in stage_contributions}
    
    fm_pct = pct_map.get("FM (First Mile)", 47.1)
    o2a_pct = pct_map.get("O2A (Order-to-Assign)", 25.3)
    wt_pct = pct_map.get("WT (Kitchen Wait Time)", 14.6)
    lm_pct = pct_map.get("LM (Last Mile)", 13.0)

    return (
        f'<div style="margin-top:14px;margin-bottom:6px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">'
        f'<div style="font-family:var(--font-heading);font-size:0.95rem;font-weight:700;color:#f8fafc;">📍 Current Order — Local SHAP Operational Attribution</div>'
        f'<div style="font-size:0.78rem;color:#94a3b8;font-family:var(--font-mono);background:rgba(15,23,42,0.8);border:1px solid rgba(255,255,255,0.08);padding:4px 10px;border-radius:9999px;">🌐 Global Stage Attribution: FM 47.1% • O2A 25.3% • WT 14.6% • LM 13.0%</div>'
        f'</div>'
        f'<div class="journey-container">'
        f'<div class="journey-node" style="border-color:rgba(129,140,248,0.35);">'
        f'<div class="journey-node-icon">📱</div>'
        f'<div class="journey-node-title">1. Order-to-Assign (O2A)</div>'
        f'<div class="journey-node-pct" style="color:#818cf8;">{o2a_pct:.1f}%</div>'
        f'<div class="journey-node-desc">Dispatch latency & rider matching</div>'
        f'</div>'
        f'<div class="journey-node" style="border-color:rgba(56,189,248,0.45);background:rgba(56,189,248,0.05);">'
        f'<div class="journey-node-icon">🚗</div>'
        f'<div class="journey-node-title">2. First Mile (FM)</div>'
        f'<div class="journey-node-pct" style="color:#38bdf8;">{fm_pct:.1f}%</div>'
        f'<div class="journey-node-desc">Transit to merchant pickup</div>'
        f'</div>'
        f'<div class="journey-node" style="border-color:rgba(245,158,11,0.35);">'
        f'<div class="journey-node-icon">🍳</div>'
        f'<div class="journey-node-title">3. Kitchen Wait (WT)</div>'
        f'<div class="journey-node-pct" style="color:#f59e0b;">{wt_pct:.1f}%</div>'
        f'<div class="journey-node-desc">Kitchen prep & food handover</div>'
        f'</div>'
        f'<div class="journey-node" style="border-color:rgba(52,211,153,0.35);">'
        f'<div class="journey-node-icon">📦</div>'
        f'<div class="journey-node-title">4. Last Mile (LM)</div>'
        f'<div class="journey-node-pct" style="color:#34d399;">{lm_pct:.1f}%</div>'
        f'<div class="journey-node-desc">Doorstep transit & dropoff</div>'
        f'</div>'
        f'</div>'
    )


def render_why_this_eta(top_delays: list[dict], top_savers: list[dict], base_value: float, predicted_eta: float) -> str:
    """Renders the 'Why this ETA?' feature attribution diagnostic card."""
    delay_html_items = []
    for item in top_delays[:3]:
        feat = item["Feature"].replace("_", " ")
        val = item["SHAP_Value"]
        delay_html_items.append(
            f'<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.05);">'
            f'<span style="color:#fca5a5;">+ {feat}</span>'
            f'<span style="font-family:var(--font-mono);font-weight:700;color:#f43f5e;">+{val:.2f} min</span>'
            f'</div>'
        )
    delay_rows = "".join(delay_html_items) if delay_html_items else '<div style="color:#64748b;">No major delay drivers</div>'

    saver_html_items = []
    for item in top_savers[:3]:
        feat = item["Feature"].replace("_", " ")
        val = item["SHAP_Value"]
        saver_html_items.append(
            f'<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.05);">'
            f'<span style="color:#86efac;">− {feat}</span>'
            f'<span style="font-family:var(--font-mono);font-weight:700;color:#10b981;">{val:.2f} min</span>'
            f'</div>'
        )
    saver_rows = "".join(saver_html_items) if saver_html_items else '<div style="color:#64748b;">No major time savers</div>'

    return (
        f'<div class="kpi-card" style="margin-top:14px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;flex-wrap:wrap;gap:8px;">'
        f'<span style="font-family:var(--font-heading);font-weight:700;font-size:1.05rem;color:#f8fafc;">🔍 Why this ETA? (SHAP Order Attribution)</span>'
        f'<span style="font-size:0.8rem;color:#94a3b8;font-family:var(--font-mono);">Base prediction: <b>{base_value:.1f} min</b> &nbsp;→&nbsp; Final prediction: <b>{predicted_eta:.1f} min</b></span>'
        f'</div>'
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;font-size:0.85rem;">'
        f'<div>'
        f'<div style="font-weight:700;color:#f87171;text-transform:uppercase;font-size:0.75rem;letter-spacing:0.05em;margin-bottom:8px;">⏳ Top Delay Drivers (+min)</div>'
        f'{delay_rows}'
        f'</div>'
        f'<div>'
        f'<div style="font-weight:700;color:#34d399;text-transform:uppercase;font-size:0.75rem;letter-spacing:0.05em;margin-bottom:8px;">⚡ Top Time Savers (−min)</div>'
        f'{saver_rows}'
        f'</div>'
        f'</div>'
        f'<div style="margin-top:12px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.06);font-size:0.76rem;color:#64748b;font-style:italic;">* Top contributing factors shown; remaining features omitted for clarity.</div>'
        f'</div>'
    )


def get_risk_badge_html(risk_label: str) -> str:
    if "Low" in risk_label:
        return f'<span class="badge-pill badge-green">🟢 {risk_label}</span>'
    elif "Medium" in risk_label:
        return f'<span class="badge-pill badge-amber">🟡 {risk_label}</span>'
    else:
        return f'<span class="badge-pill badge-red">🔴 {risk_label}</span>'


def get_risk_badge_html(risk_label: str) -> str:
    if "Low" in risk_label:
        return f'<span class="badge-pill badge-green">🟢 {risk_label}</span>'
    elif "Medium" in risk_label:
        return f'<span class="badge-pill badge-amber">🟡 {risk_label}</span>'
    else:
        return f'<span class="badge-pill badge-red">🔴 {risk_label}</span>'
