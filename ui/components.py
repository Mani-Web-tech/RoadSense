"""
RoadSense UI Components and Styling
Provides high-visibility automotive dark dashboard styling, HUD banners, and telemetry components.
"""
import streamlit as st
from typing import Dict, List, Optional
from models.risk_assessor import RiskLevel, SceneRiskSummary, ObjectRiskAssessment


def inject_custom_css() -> None:
    """Injects neon-green cyber automotive cockpit styles."""
    st.markdown("""
    <style>
    /* ══════════════════════════════════════════════════════
       GLOBAL CYBER COCKPIT CANVAS — Neon Green Theme
       ══════════════════════════════════════════════════════ */
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@700;900&display=swap');

    .stApp {
        background:
            radial-gradient(ellipse at 50% -10%, rgba(0, 255, 100, 0.12) 0%, transparent 55%),
            radial-gradient(ellipse at 10% 80%, rgba(0, 200, 80, 0.06) 0%, transparent 45%),
            linear-gradient(180deg, #001400 0%, #000d00 40%, #000800 100%);
        color: #b8ffcc;
        font-family: 'Share Tech Mono', 'Courier New', monospace;
    }

    /* Scanline overlay for CRT realism */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: repeating-linear-gradient(
            0deg,
            transparent,
            transparent 2px,
            rgba(0, 255, 100, 0.015) 2px,
            rgba(0, 255, 100, 0.015) 4px
        );
        pointer-events: none;
        z-index: 9999;
    }

    /* ── SIDEBAR ─────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #001800 0%, #000f00 100%) !important;
        border-right: 1px solid rgba(0, 255, 100, 0.2) !important;
        box-shadow: 4px 0 20px rgba(0, 255, 80, 0.08);
    }
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: #88ffaa !important;
        font-family: 'Share Tech Mono', monospace !important;
    }

    /* ── BRANDED HEADER CARD ─────────────────────────────── */
    .brand-header-card {
        background: linear-gradient(135deg,
            rgba(0, 30, 0, 0.97) 0%,
            rgba(0, 15, 0, 0.97) 100%);
        border: 1px solid rgba(0, 255, 100, 0.4);
        border-radius: 14px;
        padding: 22px 28px;
        margin-bottom: 22px;
        box-shadow:
            0 0 40px rgba(0, 255, 80, 0.15),
            0 8px 32px rgba(0, 0, 0, 0.7),
            inset 0 1px 0 rgba(0, 255, 100, 0.15);
        backdrop-filter: blur(12px);
        position: relative;
        overflow: hidden;
    }
    /* Corner accents */
    .brand-header-card::before,
    .brand-header-card::after {
        content: '';
        position: absolute;
        width: 18px; height: 18px;
        border-color: #00ff88;
        border-style: solid;
    }
    .brand-header-card::before {
        top: 6px; left: 6px;
        border-width: 2px 0 0 2px;
    }
    .brand-header-card::after {
        bottom: 6px; right: 6px;
        border-width: 0 2px 2px 0;
    }

    .brand-title {
        font-family: 'Orbitron', 'Share Tech Mono', monospace !important;
        font-size: 30px;
        font-weight: 900;
        letter-spacing: 4px;
        background: linear-gradient(90deg,
            #00ff88 0%,
            #00ffcc 40%,
            #88ffcc 70%,
            #ffffff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 14px;
        text-shadow: 0 0 30px rgba(0, 255, 136, 0.5);
        filter: drop-shadow(0 0 8px rgba(0, 255, 136, 0.6));
    }
    .brand-tagline {
        font-size: 12px;
        color: #44bb77;
        margin-top: 8px;
        letter-spacing: 2px;
        font-weight: 400;
        font-family: 'Share Tech Mono', monospace;
        text-transform: uppercase;
    }

    /* ── HUD CHIPS ───────────────────────────────────────── */
    .hud-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(0, 255, 100, 0.08);
        color: #00ff88;
        border: 1px solid rgba(0, 255, 100, 0.45);
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        font-family: 'Share Tech Mono', monospace;
        box-shadow: 0 0 8px rgba(0, 255, 100, 0.2);
        animation: chip-flicker 3s infinite;
    }
    @keyframes chip-flicker {
        0%, 95%, 100% { opacity: 1; }
        97% { opacity: 0.7; }
    }

    /* ── RISK BANNERS ────────────────────────────────────── */
    .risk-banner {
        border-radius: 8px;
        padding: 16px 22px;
        margin-bottom: 18px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 16px;
        font-family: 'Share Tech Mono', monospace;
        letter-spacing: 1px;
        position: relative;
        overflow: hidden;
    }
    .risk-banner-high {
        background: linear-gradient(90deg, #3a0000 0%, #5c0000 50%, #3a0000 100%);
        border: 2px solid #ff3333;
        color: #ffcccc;
        box-shadow: 0 0 30px rgba(255, 50, 50, 0.5), inset 0 0 20px rgba(255, 0, 0, 0.1);
        animation: pulse-high 0.9s infinite alternate;
    }
    .risk-banner-medium {
        background: linear-gradient(90deg, #2a1a00 0%, #3d2600 50%, #2a1a00 100%);
        border: 2px solid #ff9900;
        color: #ffddaa;
        box-shadow: 0 0 18px rgba(255, 153, 0, 0.4);
        animation: pulse-med 2s infinite alternate;
    }
    .risk-banner-low {
        background: linear-gradient(90deg, #001a33 0%, #002244 50%, #001a33 100%);
        border: 1px solid #0077ff;
        color: #aaccff;
        box-shadow: 0 0 10px rgba(0, 119, 255, 0.25);
    }
    .risk-banner-safe {
        background: linear-gradient(90deg, #001a00 0%, #002800 50%, #001a00 100%);
        border: 1px solid #00ff88;
        color: #aaffcc;
        box-shadow: 0 0 14px rgba(0, 255, 136, 0.2);
    }
    .risk-banner-unknown {
        background: linear-gradient(90deg, #0a0a0a 0%, #111111 50%, #0a0a0a 100%);
        border: 1px solid #334433;
        color: #668866;
    }

    @keyframes pulse-high {
        from { box-shadow: 0 0 15px rgba(255, 50, 50, 0.4), inset 0 0 10px rgba(255,0,0,0.05); }
        to   { box-shadow: 0 0 40px rgba(255, 50, 50, 0.9), inset 0 0 25px rgba(255,0,0,0.15); }
    }
    @keyframes pulse-med {
        from { box-shadow: 0 0 10px rgba(255, 153, 0, 0.3); }
        to   { box-shadow: 0 0 24px rgba(255, 153, 0, 0.7); }
    }

    /* ── METRICS ─────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: rgba(0, 20, 0, 0.6) !important;
        border: 1px solid rgba(0, 255, 100, 0.2) !important;
        border-radius: 8px !important;
        box-shadow: 0 0 12px rgba(0, 255, 80, 0.08);
        transition: box-shadow 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        box-shadow: 0 0 20px rgba(0, 255, 100, 0.2) !important;
        border-color: rgba(0, 255, 100, 0.45) !important;
    }
    [data-testid="stMetricLabel"] {
        color: #44bb77 !important;
        font-family: 'Share Tech Mono', monospace !important;
        letter-spacing: 1px;
        text-transform: uppercase;
        font-size: 11px !important;
    }
    [data-testid="stMetricValue"] {
        color: #00ff88 !important;
        font-family: 'Orbitron', monospace !important;
        font-size: 22px !important;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
    }
    [data-testid="stMetricDelta"] {
        color: #00dd66 !important;
    }

    /* ── BUTTONS ─────────────────────────────────────────── */
    [data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #003300 0%, #004d00 100%) !important;
        border: 1px solid #00ff88 !important;
        color: #00ff88 !important;
        font-family: 'Share Tech Mono', monospace !important;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        box-shadow: 0 0 12px rgba(0, 255, 136, 0.3);
        transition: all 0.2s ease;
    }
    [data-testid="stButton"] button[kind="primary"]:hover {
        background: linear-gradient(135deg, #004400 0%, #006600 100%) !important;
        box-shadow: 0 0 22px rgba(0, 255, 136, 0.6) !important;
    }
    [data-testid="stButton"] button:not([kind="primary"]) {
        background: rgba(0, 30, 0, 0.5) !important;
        border: 1px solid rgba(0, 200, 80, 0.35) !important;
        color: #88ffaa !important;
        font-family: 'Share Tech Mono', monospace !important;
        letter-spacing: 1px;
    }

    /* ── CONTAINERS / CARDS ──────────────────────────────── */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(0, 15, 0, 0.5) !important;
        border: 1px solid rgba(0, 255, 100, 0.15) !important;
        border-radius: 8px !important;
        box-shadow: 0 0 10px rgba(0, 200, 80, 0.05);
    }

    /* ── PROGRESS BAR ────────────────────────────────────── */
    [data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #00aa55 0%, #00ff88 100%) !important;
        box-shadow: 0 0 8px rgba(0, 255, 136, 0.5);
    }
    [data-testid="stProgress"] {
        background: rgba(0, 30, 0, 0.4) !important;
    }

    /* ── TABS ────────────────────────────────────────────── */
    [data-testid="stTabs"] [data-testid="stTabPanel"] {
        padding-top: 16px;
    }
    button[data-baseweb="tab"] {
        font-family: 'Share Tech Mono', monospace !important;
        letter-spacing: 1px;
        color: #44bb77 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #00ff88 !important;
        border-bottom-color: #00ff88 !important;
        text-shadow: 0 0 8px rgba(0, 255, 136, 0.6);
    }

    /* ── DATAFRAME / TABLE ───────────────────────────────── */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(0, 255, 100, 0.2) !important;
        border-radius: 8px;
        overflow: hidden;
    }

    /* ── VIDEO ELEMENT ───────────────────────────────────── */
    video {
        width: 100% !important;
        border-radius: 8px;
        border: 1px solid rgba(0, 255, 100, 0.3);
        box-shadow: 0 0 20px rgba(0, 255, 80, 0.15);
    }

    /* ── SLIDER ──────────────────────────────────────────── */
    [data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stThumb"] {
        background-color: #00ff88 !important;
        box-shadow: 0 0 8px rgba(0, 255, 136, 0.6) !important;
    }

    /* ── STEP CARDS ──────────────────────────────────────── */
    .step-card {
        background: rgba(0, 15, 0, 0.6);
        border: 1px solid rgba(0, 255, 100, 0.2);
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }

    /* ── DISCLAIMER BOX ──────────────────────────────────── */
    .disclaimer-box {
        background: rgba(0, 15, 0, 0.6);
        border-left: 3px solid #00aa55;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 12px;
        color: #44bb77;
        margin-top: 18px;
        line-height: 1.6;
        font-family: 'Share Tech Mono', monospace;
        letter-spacing: 0.5px;
    }

    /* ── SCROLLBAR ───────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #000800; }
    ::-webkit-scrollbar-thumb {
        background: #004d22;
        border-radius: 3px;
        box-shadow: 0 0 4px rgba(0,255,100,0.3);
    }
    ::-webkit-scrollbar-thumb:hover { background: #00993d; }

    /* ── METRIC ROW MIN-WIDTH ────────────────────────────── */
    [data-testid="stHorizontalBlock"] [data-testid="stMetric"] {
        min-width: 140px;
    }
    </style>
    """, unsafe_allow_html=True)


def render_header() -> None:
    """Renders the branded RoadSense top navigation header with neon-green cyber theme."""
    st.markdown("""
    <div class="brand-header-card">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
            <div class="brand-title">
                <span>🛡️ ROADSENSE</span>
            </div>
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                <span class="hud-chip">YOLOv8n + ByteTrack</span>
                <span class="hud-chip" style="background: rgba(0,255,100,0.1); color: #00ff88; border-color: rgba(0,255,100,0.5);">
                    ◉ ADAS ACTIVE
                </span>
                <span class="hud-chip" style="background: rgba(0,200,80,0.08); color: #44bb77; border-color: rgba(0,200,80,0.3);">
                    AI ONLINE
                </span>
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; margin-top: 10px; gap: 8px;">
            <div class="brand-tagline">
                // SEE THE RISK · PREDICT THE DANGER · PROTECT THE JOURNEY
            </div>
            <div style="font-size: 10px; color: #224422; font-family: 'Share Tech Mono', monospace; letter-spacing: 1px;">
                SYS OK &nbsp;|&nbsp; CPU NOMINAL &nbsp;|&nbsp; CAM LINK ACTIVE
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_risk_banner(summary: SceneRiskSummary) -> None:
    """Renders high-visibility animated status banner according to overall scene risk level."""
    lvl = summary.overall_risk_level
    reason = summary.highest_risk_reason

    if lvl == RiskLevel.HIGH:
        st.markdown(f"""
        <div class="risk-banner risk-banner-high">
            <span style="font-size: 34px;">🚨</span>
            <div style="flex-grow: 1;">
                <div style="font-size: 20px; font-weight: 800; letter-spacing: 1px;">CRITICAL COLLISION WARNING</div>
                <div style="font-size: 14px; opacity: 0.95; font-weight: 500; margin-top: 3px;">{reason}</div>
            </div>
            <div style="background: rgba(0,0,0,0.4); padding: 4px 12px; border-radius: 6px; font-size: 12px; font-weight: 700;">
                IMMEDIATE CAUTION
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif lvl == RiskLevel.MEDIUM:
        st.markdown(f"""
        <div class="risk-banner risk-banner-medium">
            <span style="font-size: 30px;">⚠️</span>
            <div style="flex-grow: 1;">
                <div style="font-size: 18px; font-weight: 700; letter-spacing: 0.8px;">CAUTION: POTENTIAL PATH CONFLICT</div>
                <div style="font-size: 13px; opacity: 0.95; font-weight: 400; margin-top: 2px;">{reason}</div>
            </div>
            <div style="background: rgba(0,0,0,0.3); padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 700;">
                CLOSING IN CORRIDOR
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif lvl == RiskLevel.LOW:
        st.markdown(f"""
        <div class="risk-banner risk-banner-low">
            <span style="font-size: 26px;">ℹ️</span>
            <div>
                <div style="font-size: 16px; font-weight: 700; letter-spacing: 0.5px;">LOW RISK — ROAD USERS MONITORED</div>
                <div style="font-size: 13px; opacity: 0.9; font-weight: 400; margin-top: 2px;">{reason}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif lvl == RiskLevel.NO_TARGETS_DETECTED:
        st.markdown(f"""
        <div class="risk-banner risk-banner-unknown">
            <span style="font-size: 26px;">👁️</span>
            <div>
                <div style="font-size: 16px; font-weight: 700; letter-spacing: 0.5px;">NO ROAD USERS DETECTED (ACTIVE MONITORING)</div>
                <div style="font-size: 12px; opacity: 0.85; font-weight: 400; margin-top: 2px;">{reason}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif lvl == RiskLevel.INSUFFICIENT_DATA:
        st.markdown(f"""
        <div class="risk-banner risk-banner-unknown">
            <span style="font-size: 26px;">⚙️</span>
            <div>
                <div style="font-size: 16px; font-weight: 700; letter-spacing: 0.5px;">INSUFFICIENT OBSERVATION DATA</div>
                <div style="font-size: 12px; opacity: 0.85; font-weight: 400; margin-top: 2px;">{reason}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="risk-banner risk-banner-safe">
            <span style="font-size: 26px;">🛡️</span>
            <div>
                <div style="font-size: 16px; font-weight: 700; letter-spacing: 0.5px;">ALL TARGETS RECEDING / NON-CONFLICTING</div>
                <div style="font-size: 12px; opacity: 0.9; font-weight: 400; margin-top: 2px;">{reason}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_safety_notice() -> None:
    """Renders formal safety boundary notice."""
    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Autonomous Safety Boundary & Research Prototype Notice:</strong><br>
        RoadSense is an experimental computer vision research prototype for Advanced Driver Assistance (ADAS) evaluation.
        All collision metrics are uncalibrated 2D image-space heuristics without physical vehicle CAN-bus telemetry
        or factory depth certification. It does NOT replace driver responsibility and does NOT actuate vehicle steering or brakes.
    </div>
    """, unsafe_allow_html=True)
