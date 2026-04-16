import html
import os
import re
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from nlp_utils import clean_text, get_sentiment, perform_topic_modeling

st.set_page_config(
    page_title="Brand Intel Studio",
    page_icon="A",
    layout="wide",
)

SAMPLE_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_comments.csv")
SOURCE_OPTIONS = (
    "Landing page",
    "Use sample data",
    "Upload CSV",
    "Fetch from YouTube",
)
THEME_OPTIONS = ("Editorial Dawn", "Campaign Night", "Sunset Pulse")


def escape_html(value):
    return html.escape(str(value))


def normalize_topic_keywords(keywords, limit=5):
    raw = str(keywords or "").strip()
    if not raw:
        return []

    parts = [piece.strip() for piece in re.split(r"[,/|]+", raw) if piece.strip()]
    if len(parts) <= 1:
        camel_parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", raw)
        if len(camel_parts) > 1:
            parts = camel_parts
        else:
            parts = re.findall(r"[A-Za-z][A-Za-z0-9_-]+", raw) or [raw]

    cleaned = []
    for piece in parts:
        piece = piece.replace("_", " ").replace("-", " ").strip()
        if piece:
            cleaned.append(piece.title())
    return cleaned[:limit]


def load_css(theme_mode):
    theme_presets = {
        "Editorial Dawn": {
            "app_bg": "radial-gradient(circle at top left, rgba(255, 107, 53, 0.20), transparent 28%), radial-gradient(circle at 86% 8%, rgba(19, 138, 114, 0.22), transparent 24%), radial-gradient(circle at 20% 82%, rgba(242, 169, 0, 0.16), transparent 20%), linear-gradient(180deg, #f7f2e8 0%, #efe7d8 100%)",
            "ink": "#11203b",
            "muted": "#5c6270",
            "sidebar_bg": "linear-gradient(180deg, rgba(11, 19, 41, 0.98) 0%, rgba(22, 32, 61, 0.96) 100%)",
            "hero_bg": "linear-gradient(135deg, rgba(255, 248, 240, 0.96) 0%, rgba(255, 255, 255, 0.80) 52%, rgba(226, 246, 241, 0.92) 100%)",
            "board_bg": "linear-gradient(180deg, rgba(17, 32, 59, 0.96) 0%, rgba(25, 37, 70, 0.95) 100%)",
            "card_bg": "rgba(255, 252, 247, 0.82)",
            "card_border": "rgba(17, 32, 59, 0.08)",
            "mini_bg": "rgba(255, 255, 255, 0.66)",
            "mini_border": "rgba(17, 32, 59, 0.08)",
            "shadow": "0 22px 60px rgba(17, 32, 59, 0.12)",
            "brand_accent": "#ff6b35",
            "brand_secondary": "#138a72",
            "brand_gold": "#f2a900",
            "section_tint": "rgba(255, 255, 255, 0.52)",
        },
        "Campaign Night": {
            "app_bg": "radial-gradient(circle at top left, rgba(255, 107, 53, 0.28), transparent 28%), radial-gradient(circle at 85% 10%, rgba(24, 197, 160, 0.22), transparent 24%), radial-gradient(circle at 18% 82%, rgba(255, 200, 87, 0.16), transparent 20%), linear-gradient(180deg, #08111f 0%, #101b2f 100%)",
            "ink": "#f5f7fb",
            "muted": "rgba(245, 247, 251, 0.72)",
            "sidebar_bg": "linear-gradient(180deg, rgba(5, 10, 19, 0.98) 0%, rgba(11, 18, 31, 0.96) 100%)",
            "hero_bg": "linear-gradient(135deg, rgba(13, 22, 38, 0.96) 0%, rgba(20, 30, 50, 0.92) 52%, rgba(12, 62, 53, 0.82) 100%)",
            "board_bg": "linear-gradient(180deg, rgba(9, 16, 28, 0.98) 0%, rgba(20, 30, 50, 0.96) 100%)",
            "card_bg": "rgba(13, 22, 38, 0.82)",
            "card_border": "rgba(255, 255, 255, 0.10)",
            "mini_bg": "rgba(255, 255, 255, 0.08)",
            "mini_border": "rgba(255, 255, 255, 0.10)",
            "shadow": "0 22px 60px rgba(0, 0, 0, 0.28)",
            "brand_accent": "#ff8a5b",
            "brand_secondary": "#25c79b",
            "brand_gold": "#ffcb57",
            "section_tint": "rgba(255, 255, 255, 0.06)",
        },
        "Sunset Pulse": {
            "app_bg": "radial-gradient(circle at top left, rgba(255, 122, 78, 0.22), transparent 24%), radial-gradient(circle at 82% 12%, rgba(255, 201, 87, 0.18), transparent 26%), radial-gradient(circle at 18% 84%, rgba(126, 87, 194, 0.10), transparent 22%), linear-gradient(180deg, #fff5ef 0%, #f7ece2 100%)",
            "ink": "#1a1b2e",
            "muted": "#645b73",
            "sidebar_bg": "linear-gradient(180deg, rgba(38, 22, 58, 0.98) 0%, rgba(57, 31, 81, 0.96) 100%)",
            "hero_bg": "linear-gradient(135deg, rgba(255, 246, 239, 0.96) 0%, rgba(255, 255, 255, 0.82) 45%, rgba(255, 229, 212, 0.92) 100%)",
            "board_bg": "linear-gradient(180deg, rgba(39, 23, 58, 0.96) 0%, rgba(71, 36, 98, 0.95) 100%)",
            "card_bg": "rgba(255, 252, 247, 0.86)",
            "card_border": "rgba(26, 27, 46, 0.08)",
            "mini_bg": "rgba(255, 255, 255, 0.70)",
            "mini_border": "rgba(26, 27, 46, 0.08)",
            "shadow": "0 24px 60px rgba(26, 27, 46, 0.14)",
            "brand_accent": "#ff7a4e",
            "brand_secondary": "#7e57c2",
            "brand_gold": "#ffb703",
            "section_tint": "rgba(255, 255, 255, 0.54)",
        },
    }
    theme = theme_presets.get(theme_mode, theme_presets["Editorial Dawn"])
    app_ink = theme["ink"]
    muted = theme["muted"]
    card_panel = theme["card_bg"]
    border_soft = theme["card_border"]
    panel_shadow = theme["shadow"]
    css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Space+Grotesk:wght@400;500;700&display=swap');

        :root {
            --bg: #f4efe5;
            --ink: __INK__;
            --panel: __PANEL__;
            --muted: __MUTED__;
            --accent: __ACCENT__;
            --accent-2: __ACCENT2__;
            --gold: __GOLD__;
            --line: __LINE__;
            --shadow: __SHADOW__;
        }

        .stApp {
            background: __APP_BG__;
            color: __APP_INK__;
        }

        [data-testid="stAppViewContainer"] {
            background: transparent;
        }

        .block-container {
            max-width: 1380px;
            padding-top: 2.2rem;
            padding-bottom: 4rem;
        }

        html,
        body,
        [class*="css"] {
            font-family: 'Space Grotesk', sans-serif;
        }

        h1,
        h2,
        h3 {
            font-family: 'Fraunces', serif !important;
            color: var(--ink) !important;
            letter-spacing: -0.03em;
        }

        p,
        li,
        label,
        span,
        div {
            color: var(--ink);
        }

        section[data-testid="stSidebar"] {
            background: __SIDEBAR_BG__;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        section[data-testid="stSidebar"] * {
            color: #f7f4ee !important;
        }

        section[data-testid="stSidebar"] .stRadio label,
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stSlider label,
        section[data-testid="stSidebar"] .stTextInput label,
        section[data-testid="stSidebar"] .stFileUploader label {
            font-weight: 600;
            letter-spacing: 0.02em;
        }

        [data-testid="stSidebar"] .stAlert {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.12);
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 999px !important;
            border: none !important;
            background: linear-gradient(135deg, #ff6b35 0%, #ff9153 100%) !important;
            color: #fff8f1 !important;
            font-weight: 700 !important;
            padding: 0.72rem 1.25rem !important;
            box-shadow: 0 14px 32px rgba(255, 107, 53, 0.26);
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 16px 34px rgba(255, 107, 53, 0.32);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.75rem;
            margin-top: 0.75rem;
            margin-bottom: 1.1rem;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 250, 242, 0.72);
            border: 1px solid rgba(17, 32, 59, 0.09);
            border-radius: 999px;
            padding: 0.55rem 1rem;
            font-weight: 600;
            color: var(--muted);
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(255, 107, 53, 0.12), rgba(19, 138, 114, 0.10)) !important;
            color: var(--ink) !important;
            border-color: rgba(255, 107, 53, 0.28) !important;
        }

        [data-testid="stMetric"] {
            background: __CARD_BG__;
            border: 1px solid __CARD_BORDER__;
            border-radius: 22px;
            padding: 1rem 1.1rem;
            box-shadow: var(--shadow);
        }

        [data-testid="stMetricLabel"] {
            font-weight: 600;
            color: var(--muted);
        }

        [data-testid="stMetricValue"] {
            color: var(--ink);
            font-family: 'Fraunces', serif;
        }

        .hero-shell {
            padding: 2.5rem;
            border-radius: 30px;
            background: __HERO_BG__;
            border: 1px solid __HERO_BORDER__;
            box-shadow: var(--shadow);
            overflow: hidden;
            position: relative;
            margin-bottom: 1.8rem;
        }

        .hero-shell::after {
            content: "";
            position: absolute;
            inset: auto -8% -28% auto;
            width: 260px;
            height: 260px;
            background: radial-gradient(circle, rgba(255, 107, 53, 0.18), transparent 65%);
            pointer-events: none;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: 1.2fr 0.92fr;
            gap: 2rem;
            align-items: center;
        }

        .eyebrow {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: rgba(17, 32, 59, 0.06);
            color: var(--ink);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 1rem;
        }

        .eyebrow-variant {
            background: rgba(255, 107, 53, 0.11);
            color: #8e2f16;
        }

        .hero-title {
            font-family: 'Fraunces', serif;
            font-size: clamp(2.9rem, 5.4vw, 5.2rem);
            line-height: 0.93;
            letter-spacing: -0.06em;
            margin: 0 0 0.9rem 0;
            max-width: 760px;
        }

        .hero-copy {
            max-width: 680px;
            font-size: 1.06rem;
            line-height: 1.7;
            color: var(--muted);
            margin-bottom: 1.15rem;
        }

        .hero-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 0.8rem;
            margin-top: 1.2rem;
            margin-bottom: 0.7rem;
        }

        .hero-badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin-top: 1.35rem;
        }

        .signal-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.55rem 0.85rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(17, 32, 59, 0.08);
            font-size: 0.92rem;
            font-weight: 600;
        }

        .signal-pill.accent {
            background: rgba(255, 107, 53, 0.11);
            border-color: rgba(255, 107, 53, 0.18);
        }

        .signal-pill.teal {
            background: rgba(19, 138, 114, 0.11);
            border-color: rgba(19, 138, 114, 0.18);
        }

        .signal-board {
            border-radius: 28px;
            padding: 1.35rem;
            background: __BOARD_BG__;
            color: #f7f4ee;
            box-shadow: 0 28px 60px rgba(17, 32, 59, 0.18);
            position: relative;
            overflow: hidden;
            animation: floaty 8s ease-in-out infinite;
        }

        .signal-board::before {
            content: "";
            position: absolute;
            inset: 0;
            background:
                radial-gradient(circle at top right, rgba(255, 107, 53, 0.16), transparent 32%),
                radial-gradient(circle at bottom left, rgba(19, 138, 114, 0.18), transparent 30%);
            pointer-events: none;
        }

        .signal-board * {
            position: relative;
            z-index: 1;
            color: #f7f4ee;
        }

        .signal-board h3 {
            font-size: 1.25rem;
            margin: 0 0 0.45rem 0;
        }

        .signal-board p {
            color: rgba(247, 244, 238, 0.78);
            margin-bottom: 0;
            line-height: 1.55;
        }

        .board-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.85rem;
            margin-top: 1rem;
        }

        .board-card {
            border-radius: 20px;
            padding: 0.95rem 1rem;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.12);
            backdrop-filter: blur(10px);
        }

        .board-card strong {
            display: block;
            font-size: 1.45rem;
            font-family: 'Fraunces', serif;
            margin-bottom: 0.2rem;
        }

        .board-card span {
            color: rgba(247, 244, 238, 0.82);
            font-size: 0.9rem;
        }

        .section-label {
            margin-top: 2.35rem;
            margin-bottom: 1rem;
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--muted);
        }

        .section-label.accent {
            color: var(--accent);
        }

        .section-label.secondary {
            color: var(--accent-2);
        }

        .feature-card,
        .story-card,
        .impact-card,
        .topic-card,
        .quote-card,
        .reco-card,
        .data-card {
            height: 100%;
            background: __CARD_BG__;
            border: 1px solid __CARD_BORDER__;
            border-radius: 24px;
            padding: 1.8rem;
            box-shadow: var(--shadow);
            position: relative;
            overflow: hidden;
            isolation: isolate;
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            color: var(--ink);
        }

        .topic-card {
            min-height: 0;
            margin-bottom: 1rem;
            padding: 1.6rem;
        }

        .feature-card::before,
        .story-card::before,
        .impact-card::before,
        .topic-card::before,
        .quote-card::before,
        .reco-card::before,
        .data-card::before {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.08), transparent 36%);
            pointer-events: none;
            z-index: 0;
        }

        .feature-card > *,
        .story-card > *,
        .impact-card > *,
        .topic-card > *,
        .quote-card > *,
        .reco-card > *,
        .data-card > * {
            position: relative;
            z-index: 1;
        }

        .feature-card h3,
        .story-card h3,
        .impact-card h3,
        .topic-card h3,
        .quote-card h3,
        .reco-card h3,
        .data-card h3 {
            margin-top: 0;
            margin-bottom: 0.85rem;
            font-size: 1.28rem;
            line-height: 1.14;
        }

        .topic-headline {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.85rem;
        }

        .topic-kicker {
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: var(--muted);
            margin-bottom: 0.35rem;
        }

        .topic-card h3 {
            margin-bottom: 0;
            font-size: 1.35rem;
            line-height: 1.15;
        }

        .topic-summary {
            margin: 0 0 1rem 0;
            color: var(--muted);
            line-height: 1.55;
        }

        .topic-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
            margin-bottom: 1.1rem;
        }

        .topic-badge {
            flex: 0 0 auto;
            padding: 0.38rem 0.65rem;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            border: 1px solid rgba(17, 32, 59, 0.08);
            background: rgba(17, 32, 59, 0.05);
        }

        .topic-badge.tone-positive {
            color: #13795b;
            background: rgba(19, 121, 91, 0.12);
            border-color: rgba(19, 121, 91, 0.18);
        }

        .topic-badge.tone-mixed {
            color: #9a6700;
            background: rgba(154, 103, 0, 0.12);
            border-color: rgba(154, 103, 0, 0.18);
        }

        .topic-badge.tone-negative {
            color: #b42318;
            background: rgba(180, 35, 24, 0.12);
            border-color: rgba(180, 35, 24, 0.18);
        }

        .feature-card p,
        .story-card p,
        .impact-card p,
        .topic-card p,
        .quote-card p,
        .reco-card p,
        .data-card p {
            color: var(--muted);
            line-height: 1.78;
            margin-bottom: 0;
        }

        .feature-card {
            position: relative;
            overflow: hidden;
            min-height: 290px;
        }

        .feature-card::after {
            content: "";
            position: absolute;
            inset: auto -10% -25% auto;
            width: 110px;
            height: 110px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(255, 107, 53, 0.16), transparent 70%);
            pointer-events: none;
        }

        .feature-kicker {
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--accent);
            margin-bottom: 0.65rem;
        }

        .feature-kicker.secondary {
            color: var(--accent-2);
        }

        .feature-kicker.gold {
            color: var(--gold);
        }

        .feature-icon {
            width: 44px;
            height: 44px;
            border-radius: 14px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: rgba(17, 32, 59, 0.06);
            margin-bottom: 1rem;
            font-size: 1.2rem;
        }

        .landing-note {
            margin-top: 1.45rem;
            padding: 1.15rem 1.2rem;
            border-radius: 20px;
            background: rgba(17, 32, 59, 0.05);
            border: 1px solid rgba(17, 32, 59, 0.08);
            color: var(--muted);
            line-height: 1.6;
        }

        .landing-note strong {
            display: block;
            margin-bottom: 0.4rem;
            color: var(--ink);
            font-size: 1rem;
        }

        .landing-note code {
            display: inline-block;
            padding: 0.18rem 0.5rem;
            margin: 0.1rem 0.15rem 0.1rem 0;
            border-radius: 999px;
            background: rgba(255, 107, 53, 0.12);
            border: 1px solid rgba(255, 107, 53, 0.18);
            color: var(--ink);
            font-weight: 700;
            font-size: 0.86rem;
        }

        .metric-card {
            border-radius: 24px;
            padding: 1.2rem;
            background: __CARD_BG__;
            border: 1px solid __CARD_BORDER__;
            box-shadow: var(--shadow);
            min-height: 152px;
            position: relative;
            overflow: hidden;
            isolation: isolate;
            backdrop-filter: blur(12px);
        }

        .metric-card::before {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.18), transparent 35%);
            pointer-events: none;
        }

        .metric-label {
            font-size: 0.9rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
        }

        .metric-value {
            font-family: 'Fraunces', serif;
            font-size: 2.2rem;
            margin: 0.4rem 0 0.55rem 0;
            line-height: 1.05;
        }

        .metric-note {
            color: var(--muted);
            line-height: 1.55;
        }

        .explain-card {
            margin-top: 0.8rem;
            padding: 1rem 1.05rem;
            border-radius: 20px;
            background: __CARD_BG__;
            border: 1px solid __CARD_BORDER__;
            color: var(--muted);
            line-height: 1.65;
        }

        .explain-card strong {
            display: block;
            margin-bottom: 0.35rem;
            color: var(--ink);
            font-size: 1rem;
        }

        .explain-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }

        .forecast-card {
            height: 100%;
            padding: 1.15rem 1.2rem;
            border-radius: 22px;
            border: 1px solid __CARD_BORDER__;
            background: __CARD_BG__;
            box-shadow: var(--shadow);
        }

        .forecast-card h4 {
            margin: 0 0 0.4rem 0;
            font-size: 1.05rem;
        }

        .forecast-card p {
            margin: 0;
            line-height: 1.65;
            color: var(--muted);
        }

        .forecast-card .forecast-value {
            font-family: 'Fraunces', serif;
            font-size: 1.65rem;
            margin: 0.35rem 0 0.5rem 0;
            color: var(--ink);
        }

        .proposal-card {
            height: 100%;
            padding: 1.15rem 1.2rem;
            border-radius: 22px;
            border: 1px solid __CARD_BORDER__;
            background: __CARD_BG__;
            box-shadow: var(--shadow);
        }

        .proposal-card h4 {
            margin: 0 0 0.4rem 0;
            font-size: 1.05rem;
        }

        .proposal-card p {
            margin: 0;
            line-height: 1.65;
            color: var(--muted);
        }

        .topic-chip {
            display: inline-flex;
            padding: 0.42rem 0.75rem;
            border-radius: 999px;
            background: rgba(17, 32, 59, 0.06);
            border: 1px solid rgba(17, 32, 59, 0.08);
            font-size: 0.83rem;
            font-weight: 700;
            line-height: 1;
        }

        .topic-example {
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(17, 32, 59, 0.08);
        }

        .topic-example strong {
            display: inline-block;
            margin-bottom: 0.35rem;
        }

        .topic-example p {
            margin: 0;
            line-height: 1.6;
        }

        .quote-card blockquote {
            margin: 0.7rem 0 0 0;
            padding-left: 1rem;
            border-left: 3px solid rgba(255, 107, 53, 0.55);
            font-size: 1rem;
            line-height: 1.7;
            color: var(--ink);
        }

        .reco-card ul {
            padding-left: 1.15rem;
            margin: 0.65rem 0 0 0;
        }

        .reco-card li {
            color: var(--muted);
            line-height: 1.55;
            margin-bottom: 0.45rem;
        }

        .tone-positive {
            color: #13795b;
            font-weight: 700;
        }

        .tone-mixed {
            color: #9a6700;
            font-weight: 700;
        }

        .tone-negative {
            color: #b42318;
            font-weight: 700;
        }

        .helper-text,
        .caption-text {
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.6;
        }

        .data-card code,
        .story-card code,
        .quote-card code,
        .reco-card code {
            display: inline-block;
            padding: 0.16rem 0.45rem;
            border-radius: 999px;
            background: rgba(255, 107, 53, 0.10);
            border: 1px solid rgba(255, 107, 53, 0.16);
            color: var(--ink);
            font-weight: 700;
        }

        div[data-testid="stCodeBlock"] {
            border-radius: 18px;
            border: 1px solid var(--line);
            background: rgba(17, 32, 59, 0.06) !important;
            padding: 0.2rem 0.6rem;
        }

        .mini-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1.35rem;
            margin-top: 1.95rem;
        }

        .schema-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 0.9rem;
            margin-top: 0.8rem;
            margin-bottom: 0.8rem;
        }

        .mini-stat {
            border-radius: 20px;
            padding: 0.95rem 1rem;
            background: __MINI_BG__;
            border: 1px solid __MINI_BORDER__;
            box-shadow: 0 12px 30px rgba(17, 32, 59, 0.08);
        }

        .mini-stat strong {
            display: block;
            font-family: 'Fraunces', serif;
            font-size: 1.8rem;
            line-height: 1;
            margin-bottom: 0.3rem;
        }

        .mini-stat span {
            color: var(--muted);
            font-size: 0.9rem;
        }

        .section-hero {
            margin: 2.5rem 0 1.2rem;
            display: flex;
            align-items: end;
            justify-content: space-between;
            gap: 1.4rem;
            flex-wrap: wrap;
        }

        .section-hero h2 {
            margin: 0;
            font-size: clamp(1.8rem, 3vw, 2.5rem);
        }

        .section-hero p {
            max-width: 640px;
            margin: 0;
            color: var(--muted);
        }

        .upload-note {
            padding: 1.1rem 1.15rem;
            border-radius: 20px;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.12), rgba(255, 255, 255, 0.06));
            border: 1px solid rgba(255, 255, 255, 0.18);
            box-shadow: 0 14px 32px rgba(0, 0, 0, 0.12);
            color: #f7f4ee;
            line-height: 1.55;
        }

        .upload-note strong {
            display: block;
            margin-bottom: 0.45rem;
            font-size: 1.02rem;
        }

        .upload-note code {
            display: inline-block;
            margin: 0.12rem 0.12rem 0.12rem 0;
            padding: 0.18rem 0.48rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.16);
            color: #ffffff;
            font-weight: 700;
            font-size: 0.86rem;
        }

        .upload-note .schema-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin-top: 0.45rem;
        }

        .upload-note .schema-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.3rem 0.55rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.10);
            font-size: 0.82rem;
            font-weight: 700;
        }

        .theme-toggle {
            margin-top: 1rem;
            padding: 1rem 1.05rem;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.12);
        }

        .visual-module {
            margin-top: 0.95rem;
            padding: 1rem 1rem 1.05rem;
            border-radius: 22px;
            background: linear-gradient(180deg, rgba(255,255,255,0.10), rgba(255,255,255,0.05));
            border: 1px solid rgba(255, 255, 255, 0.14);
            box-shadow: 0 14px 32px rgba(0, 0, 0, 0.10);
        }

        .visual-module label {
            display: block;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: rgba(255, 255, 255, 0.72);
            margin-bottom: 0.45rem;
        }

        .visual-module h4 {
            margin: 0;
            font-family: 'Fraunces', serif;
            font-size: 1.12rem;
            line-height: 1.15;
        }

        .visual-module p {
            margin: 0.45rem 0 0 0;
            color: rgba(255, 255, 255, 0.76);
            line-height: 1.55;
            font-size: 0.92rem;
        }

        .theme-preview {
            display: flex;
            gap: 0.45rem;
            margin-top: 0.85rem;
        }

        .theme-swatch {
            flex: 1 1 0;
            height: 18px;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.18);
            box-shadow: inset 0 0 0 1px rgba(0,0,0,0.04);
        }

        .theme-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.75rem;
        }

        .theme-meta span {
            display: inline-flex;
            align-items: center;
            padding: 0.28rem 0.55rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.10);
            border: 1px solid rgba(255, 255, 255, 0.10);
            font-size: 0.8rem;
            font-weight: 700;
            color: #fff8f1;
        }

        .theme-toggle label {
            font-weight: 700;
        }

        .theme-toggle small {
            color: rgba(255, 255, 255, 0.72);
        }

        .marketing-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.45rem 0.72rem;
            border-radius: 999px;
            background: rgba(255, 107, 53, 0.12);
            color: #8e2f16;
            font-weight: 700;
            font-size: 0.82rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            border-radius: 20px;
            overflow: hidden;
            border: 1px solid rgba(17, 32, 59, 0.08);
        }

        @media (max-width: 900px) {
            .block-container {
                padding-top: 1.1rem;
            }

            .hero-shell {
                padding: 1.25rem;
                border-radius: 24px;
            }

            .hero-title {
                font-size: 2.3rem;
            }

            .hero-grid,
            .mini-strip,
            .board-grid {
                grid-template-columns: 1fr;
            }
        }

        @keyframes floaty {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-6px); }
        }
        </style>
        """
    css = (
        css.replace("__INK__", app_ink)
        .replace("__PANEL__", card_panel)
        .replace("__MUTED__", muted)
        .replace("__ACCENT__", theme["brand_accent"])
        .replace("__ACCENT2__", theme["brand_secondary"])
        .replace("__GOLD__", theme["brand_gold"])
        .replace("__LINE__", border_soft)
        .replace("__SHADOW__", panel_shadow)
        .replace("__APP_BG__", theme["app_bg"])
        .replace("__APP_INK__", app_ink)
        .replace("__SIDEBAR_BG__", theme["sidebar_bg"])
        .replace(
            "__HERO_BG__",
            theme["hero_bg"],
        )
        .replace("__HERO_BORDER__", border_soft)
        .replace("__BOARD_BG__", theme["board_bg"])
        .replace("__CARD_BG__", theme["card_bg"])
        .replace("__CARD_BORDER__", theme["card_border"])
        .replace("__MINI_BG__", theme["mini_bg"])
        .replace("__MINI_BORDER__", theme["mini_border"])
    )
    st.markdown(
        css,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_sample_data():
    return pd.read_csv(SAMPLE_DATA_PATH)


def safe_ratio(numerator, denominator):
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def extract_video_id(video_input):
    video_input = str(video_input or "").strip()
    if not video_input:
        return None

    if len(video_input) == 11 and "/" not in video_input and " " not in video_input:
        return video_input

    parsed = urlparse(video_input)
    host = parsed.netloc.lower()

    if "youtu.be" in host:
        video_id = parsed.path.strip("/")
        return video_id or None

    if "youtube.com" in host:
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]

        path_parts = [part for part in parsed.path.split("/") if part]
        if len(path_parts) >= 2 and path_parts[0] in ("embed", "shorts", "live"):
            return path_parts[1]

    return None


def fetch_youtube_comments(api_key, youtube_input, max_comments=400):
    video_id = extract_video_id(youtube_input)
    if not video_id:
        return None, "I could not extract a valid YouTube video ID. Try a full watch URL, short URL, or the raw 11-character ID."

    try:
        from googleapiclient.discovery import build

        youtube = build("youtube", "v3", developerKey=api_key)
        collected = []
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText",
        )

        while request is not None and len(collected) < max_comments:
            response = request.execute()
            for item in response.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                collected.append(
                    {
                        "comment_text": snippet.get("textDisplay", ""),
                        "likes": snippet.get("likeCount", 0),
                        "published_at": snippet.get("publishedAt"),
                        "platform": "YouTube",
                        "author": snippet.get("authorDisplayName", "Anonymous"),
                        "video_id": video_id,
                    }
                )
                if len(collected) >= max_comments:
                    break

            next_token = response.get("nextPageToken")
            if next_token and len(collected) < max_comments:
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=100,
                    textFormat="plainText",
                    pageToken=next_token,
                )
            else:
                request = None

    except Exception as exc:
        error_text = str(exc)
        lowered = error_text.lower()
        if "quotaexceeded" in lowered:
            return None, "The YouTube API quota is exhausted right now. Try again later or swap in a new key."
        if "forbidden" in lowered:
            return None, "This video does not expose comments to the API, or the key does not have access."
        if "videonotfound" in lowered or "404" in lowered:
            return None, "The video was not found. Double-check the URL or ID and try again."
        return None, "YouTube returned an error: {0}".format(error_text)

    if not collected:
        return None, "No comments came back. The video may have comments disabled."

    return pd.DataFrame(collected), None


def classify_archetype(comment_text, sentiment_score):
    comment_text = str(comment_text or "")
    if "?" in comment_text:
        return "Questions"
    if sentiment_score >= 0.35:
        return "Advocates"
    if sentiment_score <= -0.25:
        return "Risk signals"
    return "Observers"


def classify_energy(sentiment_score):
    magnitude = abs(sentiment_score)
    if magnitude >= 0.55:
        return "High voltage"
    if magnitude >= 0.20:
        return "Active"
    return "Calm"


@st.cache_data(show_spinner=False)
def process_comments(df, requested_topics):
    processed = df.copy()
    processed["comment_text"] = processed["comment_text"].fillna("").astype(str)

    if "platform" not in processed.columns:
        processed["platform"] = "Unknown"
    processed["platform"] = processed["platform"].fillna("Unknown").astype(str)

    if "likes" not in processed.columns:
        processed["likes"] = 0
    processed["likes"] = pd.to_numeric(processed["likes"], errors="coerce").fillna(0)

    timestamp_col = None
    for candidate in ("timestamp", "published_at", "created_at", "date"):
        if candidate in processed.columns:
            timestamp_col = candidate
            break

    if timestamp_col:
        processed["timestamp_dt"] = pd.to_datetime(processed[timestamp_col], errors="coerce")
    else:
        processed["timestamp_dt"] = pd.NaT

    processed["clean_text"] = processed["comment_text"].apply(clean_text)

    sentiments = processed["clean_text"].apply(get_sentiment)
    processed["sentiment_score"] = [item[0] for item in sentiments]
    processed["sentiment_label"] = [item[1] for item in sentiments]
    processed["question_flag"] = processed["comment_text"].str.contains(r"\?", regex=True, na=False)
    processed["archetype"] = processed.apply(
        lambda row: classify_archetype(row["comment_text"], row["sentiment_score"]),
        axis=1,
    )
    processed["energy_band"] = processed["sentiment_score"].apply(classify_energy)

    processed["topic_cluster"] = -1
    topic_keywords = []
    actual_topics = 0

    valid_mask = processed["clean_text"].str.strip().ne("")
    valid_texts = processed.loc[valid_mask, "clean_text"]

    if len(valid_texts) >= 4:
        topic_target = max(2, min(int(requested_topics), len(valid_texts) - 1))
        clusters, topic_keywords, _ = perform_topic_modeling(
            valid_texts.tolist(), n_topics=topic_target
        )
        if clusters is not None and topic_keywords is not None:
            processed.loc[valid_mask, "topic_cluster"] = clusters
            actual_topics = len(topic_keywords)

    return processed, topic_keywords, actual_topics


def keyword_frequency(text_series, limit=12):
    tokens = []
    for text in text_series.fillna("").astype(str):
        cleaned = clean_text(text)
        tokens.extend(
            token
            for token in cleaned.split()
            if token not in ENGLISH_STOP_WORDS and len(token) > 2
        )

    if not tokens:
        return pd.DataFrame(columns=["keyword", "count"])

    counts = pd.Series(tokens).value_counts().head(limit)
    return counts.rename_axis("keyword").reset_index(name="count")


def build_topic_summaries(df, topic_keywords):
    total_comments = max(len(df), 1)
    summaries = []

    for index, keywords in enumerate(topic_keywords):
        topic_df = df[df["topic_cluster"] == index].copy()
        if topic_df.empty:
            continue

        avg_score = topic_df["sentiment_score"].mean()
        if avg_score >= 0.12:
            tone_text = "Positive pull"
            tone_class = "tone-positive"
        elif avg_score <= -0.12:
            tone_text = "Friction point"
            tone_class = "tone-negative"
        else:
            tone_text = "Mixed room"
            tone_class = "tone-mixed"

        examples = (
            topic_df.sort_values(["likes", "sentiment_score"], ascending=[False, False])["comment_text"]
            .head(2)
            .tolist()
        )

        summaries.append(
            {
                "id": index,
                "keywords": keywords,
                "count": len(topic_df),
                "share": len(topic_df) / float(total_comments),
                "avg_score": avg_score,
                "tone_text": tone_text,
                "tone_class": tone_class,
                "examples": examples,
            }
        )

    return sorted(summaries, key=lambda item: item["count"], reverse=True)


def pick_quotes(df, sentiment_label, limit=2):
    subset = df[df["sentiment_label"] == sentiment_label].copy()
    if subset.empty:
        return []

    sentiment_ascending = sentiment_label == "Negative"
    subset = subset.sort_values(
        ["likes", "sentiment_score"],
        ascending=[False, sentiment_ascending],
    )
    return subset["comment_text"].head(limit).tolist()


def build_headline_metrics(df, topic_summaries):
    total = len(df)
    positive_share = safe_ratio((df["sentiment_label"] == "Positive").sum(), total)
    neutral_share = safe_ratio((df["sentiment_label"] == "Neutral").sum(), total)
    negative_share = safe_ratio((df["sentiment_label"] == "Negative").sum(), total)
    question_share = safe_ratio(df["question_flag"].sum(), total)
    avg_sentiment = df["sentiment_score"].mean() if total else 0.0
    avg_likes = df["likes"].mean() if total else 0.0

    health_score = int(
        np.clip(
            58
            + (positive_share - negative_share) * 54
            + avg_sentiment * 18
            + min(avg_likes / 14.0, 1.0) * 7
            - question_share * 6,
            0,
            100,
        )
    )

    if health_score >= 78:
        heat_note = "Strong resonance. The audience is leaning in."
    elif health_score >= 62:
        heat_note = "Healthy pull with room to sharpen."
    elif health_score >= 48:
        heat_note = "Mixed reception. Message tuning will help."
    else:
        heat_note = "Repair mode. Friction is outrunning affinity."

    dominant_value = "Topic engine warming up"
    dominant_note = "Add more comments for stronger clustering."
    if topic_summaries:
        dominant_topic = topic_summaries[0]
        dominant_value = ", ".join(normalize_topic_keywords(dominant_topic["keywords"], limit=2))
        dominant_note = "{0} comments are orbiting this theme.".format(
            dominant_topic["count"]
        )

    metrics = [
        {
            "label": "Brand heat",
            "value": "{0}/100".format(health_score),
            "note": heat_note,
        },
        {
            "label": "Positive lift",
            "value": "{0:.0f}%".format(positive_share * 100),
            "note": "{0:.0f}% negative and {1:.0f}% neutral.".format(
                negative_share * 100,
                neutral_share * 100,
            ),
        },
        {
            "label": "Question load",
            "value": "{0:.0f}%".format(question_share * 100),
            "note": "Open loops the brand should answer fast.",
        },
        {
            "label": "Dominant theme",
            "value": dominant_value,
            "note": dominant_note,
        },
    ]

    pulse = {
        "positive_share": positive_share,
        "neutral_share": neutral_share,
        "negative_share": negative_share,
        "question_share": question_share,
        "health_score": health_score,
    }
    return metrics, pulse


def build_recommendations(df, topic_summaries):
    recommendations = {"Act now": [], "Amplify": [], "Watch next": []}
    negative_topics = [topic for topic in topic_summaries if topic["avg_score"] <= -0.05]
    positive_topics = [topic for topic in topic_summaries if topic["avg_score"] >= 0.05]

    question_share = safe_ratio(df["question_flag"].sum(), len(df))

    if negative_topics:
        lead = negative_topics[0]
        lead_topic = ", ".join(normalize_topic_keywords(lead["keywords"], limit=3))
        recommendations["Act now"].append(
            "Repair the {0} narrative. {1} comments in this cluster lean negative, so answer it with proof, pricing context, or support content.".format(
                lead_topic,
                lead["count"],
            )
        )
    else:
        recommendations["Act now"].append(
            "Negative signal is limited right now. Keep monitoring for early warning shifts after each campaign launch."
        )

    if question_share >= 0.18:
        recommendations["Act now"].append(
            "Question volume is elevated. Build a fast-answer content block for recurring concerns so curiosity does not turn into drop-off."
        )
    else:
        recommendations["Act now"].append(
            "Audience confusion is manageable. Keep FAQs tight and use top questions as copy prompts for the next launch burst."
        )

    if positive_topics:
        win = positive_topics[0]
        win_topic = ", ".join(normalize_topic_keywords(win["keywords"], limit=3))
        recommendations["Amplify"].append(
            "Turn {0} into headline language. This topic shows the clearest positive pull in the conversation.".format(
                win_topic
            )
        )

    positive_likes = df.loc[df["sentiment_label"] == "Positive", "likes"].mean()
    negative_likes = df.loc[df["sentiment_label"] == "Negative", "likes"].mean()
    if positive_likes >= negative_likes:
        recommendations["Amplify"].append(
            "Positive comments carry stronger engagement than negative ones. Recycle that language into testimonials, hooks, and pinned replies."
        )
    else:
        recommendations["Amplify"].append(
            "Negative comments are drawing more interaction. Address them publicly with confident, specific responses before amplifying the positives."
        )

    recommendations["Watch next"].append(
        "Keep a weekly snapshot of sentiment, topic mix, and question load so you can spot messaging drift before the audience feels it."
    )
    if topic_summaries:
        dominant = ", ".join(normalize_topic_keywords(topic_summaries[0]["keywords"], limit=3))
        recommendations["Watch next"].append(
            "Your largest conversation cluster is {0}. Track whether it stays opportunity-led or slides into friction over time.".format(
                dominant
            )
        )

    return recommendations


def build_future_insights(df, topic_summaries, pulse):
    insights = []
    total = len(df)

    if total == 0:
        return insights

    if pulse["question_share"] >= 0.18:
        insights.append(
            {
                "title": "Support and FAQ demand",
                "value": "High",
                "note": "Open questions are likely to keep rising unless the brand creates a fast-answer content block.",
                "proposal": "Turn the top 5 recurring questions into a pinned FAQ, short video, or landing-page explainer.",
            }
        )
    else:
        insights.append(
            {
                "title": "Support and FAQ demand",
                "value": "Stable",
                "note": "Question load is under control, so the next move is to preserve clarity and watch for new objections.",
                "proposal": "Keep a weekly question review to catch new friction before it becomes a pattern.",
            }
        )

    if topic_summaries:
        top_topic = topic_summaries[0]
        top_label = ", ".join(normalize_topic_keywords(top_topic["keywords"], limit=3)) or "Top topic"
        if top_topic["avg_score"] >= 0.08:
            insights.append(
                {
                    "title": "Content opportunity",
                    "value": "Push",
                    "note": "The strongest topic is pulling positive attention, which makes it a good candidate for testimonials, hooks, and campaign messaging.",
                    "proposal": "Turn {0} into headline copy, ad language, and pinned reply material while the signal is warm.".format(top_label),
                }
            )
        else:
            insights.append(
                {
                    "title": "Content opportunity",
                    "value": "Repair",
                    "note": "The top conversation lane has mixed energy, so the brand should answer it with more clarity and proof.",
                    "proposal": "Create a response asset around {0} and address the concern with examples, pricing context, or support detail.".format(top_label),
                }
            )

    if pulse["positive_share"] >= pulse["negative_share"]:
        insights.append(
            {
                "title": "Brand momentum",
                "value": "Upward",
                "note": "Positive language currently outpaces friction, so there is room to scale the current message instead of rewriting it.",
                "proposal": "Reuse audience phrases in social proof, ad headlines, and email openers to keep the momentum visible.",
            }
        )
    else:
        insights.append(
            {
                "title": "Brand momentum",
                "value": "Needs repair",
                "note": "Negative pressure is slightly stronger, so the next campaign should prioritize clarity and trust building.",
                "proposal": "Lead with a clearer offer, a stronger guarantee, or a proof-led explainer before amplifying reach.",
            }
        )

    if "timestamp_dt" in df.columns and df["timestamp_dt"].notna().sum() >= 4:
        timeline = (
            df.dropna(subset=["timestamp_dt"])
            .set_index("timestamp_dt")
            .resample("D")
            .agg(avg_sentiment=("sentiment_score", "mean"), comment_volume=("comment_text", "count"))
        )
        if len(timeline) >= 4:
            timeline = timeline.reset_index()
            x = np.arange(len(timeline))
            trend = np.polyfit(x, timeline["avg_sentiment"].fillna(0).to_numpy(), 1)[0]
            if trend > 0.01:
                forecast_value = "Improving"
                forecast_note = "Sentiment is trending upward over time, so the next week is a good window for a louder marketing push."
                forecast_proposal = "Schedule a new launch burst, because the audience is already warming to the message."
            elif trend < -0.01:
                forecast_value = "Softening"
                forecast_note = "Sentiment is easing down, which suggests the message should be refreshed before the next push."
                forecast_proposal = "Refresh the framing, shorten the copy, and answer the top objections before scaling spend."
            else:
                forecast_value = "Flat"
                forecast_note = "The sentiment line is steady, so the next win will come from sharper creative rather than more volume."
                forecast_proposal = "Test two new hooks and compare which one lifts positive reaction fastest."

            insights.append(
                {
                    "title": "7-day signal outlook",
                    "value": forecast_value,
                    "note": forecast_note,
                    "proposal": forecast_proposal,
                }
            )

    return insights


def render_explanation(title, body, action=None):
    st.markdown(
        """
        <div class="explain-card">
            <strong>{title}</strong>
            <div>{body}</div>
            {action}
        </div>
        """.format(
            title=escape_html(title),
            body=escape_html(body),
            action=(
                "<div style='margin-top:0.6rem; font-weight:700; color: var(--ink);'>Next: {0}</div>".format(
                    escape_html(action)
                )
                if action
                else ""
            ),
        ),
        unsafe_allow_html=True,
    )


def render_metric_cards(metrics):
    columns = st.columns(len(metrics))
    for column, metric in zip(columns, metrics):
        column.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
            """.format(
                label=escape_html(metric["label"]),
                value=escape_html(metric["value"]),
                note=escape_html(metric["note"]),
            ),
            unsafe_allow_html=True,
        )


def render_quote_card(title, description, quotes):
    if quotes:
        quote_html = "".join(
            "<blockquote>{0}</blockquote>".format(escape_html(quote)) for quote in quotes
        )
    else:
        quote_html = (
            "<blockquote>No strong signal surfaced yet. Add more comments or loosen the filters to reveal clearer language.</blockquote>"
        )

    st.markdown(
        """
        <div class="quote-card">
            <h3>{title}</h3>
            <p>{description}</p>
            {quote_html}
        </div>
        """.format(
            title=escape_html(title),
            description=escape_html(description),
            quote_html=quote_html,
        ),
        unsafe_allow_html=True,
    )


def render_topic_card(summary):
    keywords = normalize_topic_keywords(summary["keywords"], limit=5)
    keyword_markup = "".join(
        "<span class='topic-chip'>{0}</span>".format(escape_html(keyword))
        for keyword in keywords
    )
    primary_label = ", ".join(keywords[:3]) if keywords else "Unclear lane"

    example_markup = "".join(
        "<div class='topic-example'><p><strong>Signal:</strong> {0}</p></div>".format(
            escape_html(example)
        )
        for example in summary["examples"]
    )

    st.markdown(
        """
        <div class="topic-card">
            <div class="topic-headline">
                <div>
                    <div class="topic-kicker">Topic {topic_number}</div>
                    <h3>{primary_label}</h3>
                </div>
                <div class="topic-badge {tone_class}">{tone_text}</div>
            </div>
            <p class="topic-summary">{count} comments, {share}% of the conversation.</p>
            <div class="topic-chip-row">{keyword_markup}</div>
            {example_markup}
        </div>
        """.format(
            topic_number=summary["id"] + 1,
            tone_class=summary["tone_class"],
            tone_text=escape_html(summary["tone_text"]),
            count=summary["count"],
            share=int(round(summary["share"] * 100)),
            keyword_markup=keyword_markup,
            primary_label=escape_html(primary_label),
            example_markup=example_markup,
        ),
        unsafe_allow_html=True,
    )


def render_recommendation_card(title, items):
    item_markup = "".join("<li>{0}</li>".format(escape_html(item)) for item in items)
    st.markdown(
        """
        <div class="reco-card">
            <h3>{title}</h3>
            <ul>{items}</ul>
        </div>
        """.format(title=escape_html(title), items=item_markup),
        unsafe_allow_html=True,
    )


def render_algorithm_card(title, subtitle, body, bullets):
    bullet_markup = "".join(
        "<li>{0}</li>".format(escape_html(item)) for item in bullets
    )
    st.markdown(
        """
        <div class="proposal-card">
            <div class="feature-kicker gold">{subtitle}</div>
            <h4>{title}</h4>
            <p>{body}</p>
            <ul style="margin:0.9rem 0 0 1.1rem; color: var(--muted); line-height:1.6;">{bullets}</ul>
        </div>
        """.format(
            title=escape_html(title),
            subtitle=escape_html(subtitle),
            body=escape_html(body),
            bullets=bullet_markup,
        ),
        unsafe_allow_html=True,
    )


def set_data_source(mode):
    st.session_state["data_source"] = mode


def render_landing_page(mode_hint):
    sample_df = load_sample_data()
    sample_comments = len(sample_df)
    sample_platforms = sample_df["platform"].nunique() if "platform" in sample_df.columns else 0
    sample_questions = int(
        sample_df["comment_text"].fillna("").str.contains(r"\?", regex=True).sum()
    )
    avg_likes = int(round(sample_df["likes"].fillna(0).mean())) if "likes" in sample_df.columns else 0
    positive_share = int(
        round(
            sample_df["comment_text"]
            .fillna("")
            .str.contains(r"(?i)love|great|amazing|happy|recommend", regex=True)
            .mean()
            * 100
        )
    )

    helper_copy = {
        "Landing page": "Start with the sample campaign or connect a real conversation stream when you are ready.",
        "Upload CSV": "Your campaign cockpit is ready. Add a CSV in the sidebar and turn raw responses into a story you can act on.",
        "Fetch from YouTube": "Paste a video URL and API key in the sidebar to transform audience replies into message strategy.",
        "Use sample data": "The demo dataset is one click away if you want to experience the full product flow first.",
    }.get(mode_hint, "Load a dataset from the sidebar to start the analysis.")

    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-grid">
                <div>
                    <div class="marketing-badge">Marketing intelligence studio</div>
                    <div class="eyebrow eyebrow-variant">Brand Intel Studio</div>
                    <div class="hero-title">Turn audience reaction into a campaign-ready growth story.</div>
                    <p class="hero-copy">This studio helps brands see what is landing, what is confusing, and what should be amplified next. It turns comment streams into a marketing signal with message clarity, resonance, and strategic next steps.</p>
                    <p class="hero-copy">{helper_copy}</p>
                    <div class="hero-actions">
                        <span class="signal-pill accent">Track message resonance</span>
                        <span class="signal-pill teal">Decode campaign feedback</span>
                        <span class="signal-pill">Plan the next move</span>
                    </div>
                    <div class="hero-badge-row">
                        <span class="signal-pill">{sample_comments} demo comments</span>
                        <span class="signal-pill">{sample_platforms} channels represented</span>
                        <span class="signal-pill">{sample_questions} audience questions</span>
                    </div>
                </div>
                <div class="signal-board">
                    <div class="eyebrow" style="background: rgba(255,255,255,0.08); color: #fff8f1;">Live signal board</div>
                    <h3>Campaign pulse at a glance</h3>
                    <p>Use the sample dataset to preview how the market is responding before you connect live comments.</p>
                    <div class="board-grid">
                        <div class="board-card">
                            <strong>{sample_comments}</strong>
                            <span>comments in the demo set</span>
                        </div>
                        <div class="board-card">
                            <strong>{avg_likes}</strong>
                            <span>average engagement</span>
                        </div>
                        <div class="board-card">
                            <strong>{positive_share}%</strong>
                            <span>positive language estimate</span>
                        </div>
                        <div class="board-card">
                            <strong>{sample_platforms}</strong>
                            <span>platforms already covered</span>
                        </div>
                    </div>
                    <div style="margin-top:1rem; border-top:1px solid rgba(255,255,255,0.12); padding-top:0.85rem;">
                        <p style="margin:0;"><strong style="font-family:'Fraunces', serif; font-size:1.15rem;">Signal to action</strong></p>
                        <p style="margin-top:0.35rem;">The studio surfaces what people love, what they question, and where the message needs to be sharpened next.</p>
                    </div>
                </div>
            </div>
            <div class="mini-strip">
                <div class="mini-stat">
                    <strong>01</strong>
                    <span>Capture the conversation</span>
                </div>
                <div class="mini-stat">
                    <strong>02</strong>
                    <span>Decode sentiment and topics</span>
                </div>
                <div class="mini-stat">
                    <strong>03</strong>
                    <span>Reveal actionable strategy</span>
                </div>
                <div class="mini-stat">
                    <strong>04</strong>
                    <span>Export insights and respond</span>
                </div>
            </div>
        </div>
        """.format(
            helper_copy=escape_html(helper_copy),
            sample_comments=sample_comments,
            sample_platforms=sample_platforms,
            sample_questions=sample_questions,
            avg_likes=avg_likes,
            positive_share=positive_share,
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-label accent" style="margin-top:2.45rem;">What this project does</div>
        <div class="section-hero" style="margin-top:0; margin-bottom:1rem;">
            <div>
                <h2>Explainable brand intelligence for future marketing decisions</h2>
                <p>This project takes comment data and turns it into sentiment signals, topic clusters, future insights, and marketing proposals. It is built to help brands understand what happened, what is happening now, and what to do next.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    overview_columns = st.columns(3, gap="large")
    overview_cards = [
        (
            "Real-time analysis",
            "The app reads comment streams and scores the audience reaction so teams can see whether the campaign is landing or drifting.",
            "Sentiment + tone",
            ["TextBlob sentiment scoring", "Audience posture detection", "Question signal tracking"],
        ),
        (
            "Explainable clustering",
            "It groups comments into narrative lanes and shows the words, signals, and examples that define each cluster.",
            "TF-IDF + KMeans",
            ["Topic discovery with TF-IDF", "KMeans clustering", "Readable topic labels"],
        ),
        (
            "Future marketing ideas",
            "The strategy layer proposes next moves, content opportunities, and forecast-style insights so the output is usable for planning.",
            "Forecast + proposal",
            ["Trend outlook", "Campaign proposals", "FAQ and content ideas"],
        ),
    ]
    for column, (title, body, subtitle, bullets) in zip(overview_columns, overview_cards):
        with column:
            render_algorithm_card(title, subtitle, body, bullets)

    action_col, note_col = st.columns([0.28, 0.72])
    with action_col:
        st.button(
            "Launch sample cockpit",
            width="stretch",
            on_click=set_data_source,
            args=("Use sample data",),
        )
    with note_col:
        st.markdown(
            "<p class='caption-text'>Use the sidebar to upload a CSV, connect YouTube comments, or stay on the landing page while you explore the product story.</p>",
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="landing-note">
            <strong>Data contract</strong>
            <div class="schema-strip">
                <span class="signal-pill accent">Required: comment_text</span>
                <span class="signal-pill">Best extra: platform</span>
                <span class="signal-pill">Best extra: likes</span>
                <span class="signal-pill teal">Best extra: timestamp</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-hero">
            <div>
                <h2>Built for marketing teams who need signal, not noise</h2>
                <p>The page now frames the product as a campaign intelligence studio with a clear value proposition: understand audience reaction, sharpen the message, and turn comments into action.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    feature_columns = st.columns(3, gap="large")
    feature_cards = [
        (
            "Audience lens",
            "See what people are saying, what they are asking, and which message is winning attention before the campaign loses momentum.",
            "01",
            "Reach and resonance",
        ),
        (
            "Campaign clarity",
            "The studio translates comments into a marketing-friendly readout so strategy, content, and support teams can move faster together.",
            "02",
            "Message testing",
        ),
        (
            "Decision ready",
            "Live demo, upload, and YouTube paths stay visible from the start so the product feels usable, not hidden behind settings.",
            "03",
            "Always usable",
        ),
    ]
    for column, (title, copy, icon, kicker) in zip(feature_columns, feature_cards):
        column.markdown(
            """
            <div class="feature-card">
                <div class="feature-kicker">{kicker}</div>
                <div class="feature-icon">{icon}</div>
                <h3>{title}</h3>
                <p>{copy}</p>
            </div>
            """.format(
                title=escape_html(title),
                copy=escape_html(copy),
                icon=escape_html(icon),
                kicker=escape_html(kicker),
            ),
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="section-label secondary" style="margin-top:2.6rem;">How it works</div>
        <div class="section-hero" style="margin-top:0; margin-bottom:1.2rem;">
            <div>
                <h2>From social reactions to campaign moves</h2>
                <p>Each step is separated so the product story reads like a premium workflow instead of one packed block.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    step_columns = st.columns(3, gap="large")
    step_cards = [
        (
            "01",
            "Ingest",
            "Bring in a CSV, load the demo, or connect a YouTube campaign.",
        ),
        (
            "02",
            "Decode",
            "The app scores sentiment, detects audience questions, and groups the conversation into narrative lanes.",
        ),
        (
            "03",
            "Direct",
            "Strategy cards tell the team what to repair, what to amplify, and what to monitor next.",
        ),
    ]
    for column, (step, title, copy) in zip(step_columns, step_cards):
        column.markdown(
            """
            <div class="feature-card">
                <div class="feature-kicker secondary">Step {step}</div>
                <div class="feature-icon">{step}</div>
                <h3>{title}</h3>
                <p>{copy}</p>
            </div>
            """.format(
                step=escape_html(step),
                title=escape_html(title),
                copy=escape_html(copy),
            ),
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="section-label accent" style="margin-top:2.4rem;">Input shape</div>
        <div class="data-card">
            <div class="feature-kicker gold">What the app expects</div>
            <h3>Required and helpful columns</h3>
            <p>Required: <code>comment_text</code></p>
            <div class="schema-strip" style="margin-top:1rem; margin-bottom:1rem;">
                <span class="signal-pill">platform</span>
                <span class="signal-pill">likes</span>
                <span class="signal-pill teal">timestamp</span>
                <span class="signal-pill accent">published_at</span>
            </div>
            <p>If those extra columns are missing, the app still works and fills in smart defaults.</p>
        </div>
        <div class="landing-note">
            <strong>Quick reference</strong>
            <div class="schema-strip">
                <span class="signal-pill accent">comment_text</span>
                <span class="signal-pill">platform</span>
                <span class="signal-pill">likes</span>
                <span class="signal-pill teal">timestamp</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard(df, topic_keywords, actual_topics, source_label):
    topic_summaries = build_topic_summaries(df, topic_keywords)
    metrics, pulse = build_headline_metrics(df, topic_summaries)
    platform_count = df["platform"].nunique() if "platform" in df.columns else 1

    dominant_theme = "Narratives are still forming"
    if topic_summaries:
        dominant_theme = ", ".join(topic_summaries[0]["keywords"].split(", ")[:2]).title()

    st.markdown(
        """
        <div class="hero-shell">
            <div class="eyebrow">{source_label}</div>
            <div class="hero-title">Your audience is telling you what to say next.</div>
            <p class="hero-copy">We translated {comment_count} comments across {platform_count} platform lanes into emotional signal, narrative clusters, and strategic action cues so the next brand move can be faster and sharper.</p>
            <div class="hero-badge-row">
                <span class="signal-pill">{positive_share:.0f}% positive lift</span>
                <span class="signal-pill">{question_share:.0f}% question load</span>
                <span class="signal-pill">{dominant_theme}</span>
            </div>
        </div>
        """.format(
            source_label=escape_html(source_label),
            comment_count=len(df),
            platform_count=platform_count,
            positive_share=pulse["positive_share"] * 100,
            question_share=pulse["question_share"] * 100,
            dominant_theme=escape_html(dominant_theme),
        ),
        unsafe_allow_html=True,
    )

    action_col_1, action_col_2, action_col_3 = st.columns([0.25, 0.22, 0.53])
    with action_col_1:
        st.download_button(
            "Download enriched CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="brand_intel_studio_analysis.csv",
            mime="text/csv",
            width="stretch",
        )
    with action_col_2:
        st.button(
            "Return to landing",
            width="stretch",
            on_click=set_data_source,
            args=("Landing page",),
        )
    with action_col_3:
        st.markdown(
            "<p class='helper-text'>Use the sidebar to tune platforms, likes, keywords, and the number of narrative clusters.</p>",
            unsafe_allow_html=True,
        )

    render_metric_cards(metrics)

    mission_tab, signal_tab, topics_tab, language_tab, algorithm_tab, strategy_tab = st.tabs(
        [
            "Mission Control",
            "Signal Deep Dive",
            "Topic Radar",
            "Language Lab",
            "Algorithm Lab",
            "Strategy Studio",
        ]
    )

    with mission_tab:
        left_col, right_col = st.columns([1.15, 0.85])
        with left_col:
            st.markdown("<div class='section-label'>Conversation mix</div>", unsafe_allow_html=True)
            sentiment_counts = (
                df["sentiment_label"]
                .value_counts()
                .reindex(["Positive", "Neutral", "Negative"])
                .fillna(0)
            )
            st.bar_chart(sentiment_counts)
            render_explanation(
                "What this chart means",
                "This distribution shows the emotional balance of the conversation. More positive bars mean the message is landing well; more negative bars means the brand should tighten the offer or clarify the story.",
                "If negative sentiment grows, lead with proof, clarity, and support content.",
            )

            timeline_df = df.dropna(subset=["timestamp_dt"]).copy()
            if len(timeline_df) >= 2:
                trend = (
                    timeline_df.set_index("timestamp_dt")
                    .resample("D")
                    .agg(
                        average_sentiment=("sentiment_score", "mean"),
                        comment_volume=("comment_text", "count"),
                    )
                )
                st.markdown("<div class='section-label'>Timeline pulse</div>", unsafe_allow_html=True)
                st.line_chart(trend[["average_sentiment"]])
                st.bar_chart(trend[["comment_volume"]])
                render_explanation(
                    "Why this matters",
                    "The line shows whether sentiment is improving or declining over time, while the bars show whether attention is rising. Together, they tell you if the campaign is getting stronger or simply louder.",
                    "Use the slope to decide whether to scale the current message or refresh it.",
                )
            else:
                st.markdown(
                    """
                    <div class="data-card">
                        <h3>Timeline view is waiting for richer timestamps</h3>
                        <p>This dataset does not have enough valid time points yet, so the command center is focusing on overall message signal instead.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with right_col:
            st.markdown("<div class='section-label'>Audience posture</div>", unsafe_allow_html=True)
            archetype_counts = df["archetype"].value_counts()
            st.bar_chart(archetype_counts)
            render_explanation(
                "Audience posture",
                "This chart shows how people are behaving in the conversation: advocates are helping the brand, observers are just watching, and questions or risk signals show where attention needs a response.",
                "If questions or risk signals grow, answer them before the next campaign push.",
            )

            st.markdown("<div class='section-label'>Platform spread</div>", unsafe_allow_html=True)
            platform_counts = df["platform"].fillna("Unknown").value_counts()
            st.bar_chart(platform_counts)
            render_explanation(
                "Platform spread",
                "This view shows where the conversation is happening. A healthy mix means the message travels well; a single dominant platform means you should tailor creative to that channel's format.",
                "Repurpose the strongest language for the busiest platform first.",
            )

            top_rows = (
                df.sort_values(["likes", "sentiment_score"], ascending=[False, False])
                .loc[:, ["comment_text", "platform", "likes", "sentiment_label"]]
                .head(6)
            )
            st.markdown("<div class='section-label'>High-signal comments</div>", unsafe_allow_html=True)
            st.dataframe(top_rows, width="stretch", hide_index=True)

    with signal_tab:
        quote_left, quote_right = st.columns(2)
        with quote_left:
            render_quote_card(
                "Positive language worth amplifying",
                "These comments are the closest thing to ready-made brand proof.",
                pick_quotes(df, "Positive", limit=3),
            )
        with quote_right:
            render_quote_card(
                "Negative language to resolve",
                "These comments tell us where trust is leaking or where the offer needs clarity.",
                pick_quotes(df, "Negative", limit=3),
            )

        detail_left, detail_right = st.columns([0.92, 1.08])
        with detail_left:
            st.markdown("<div class='section-label'>Emotion intensity</div>", unsafe_allow_html=True)
            energy_counts = df["energy_band"].value_counts().reindex(
                ["High voltage", "Active", "Calm"]
            ).fillna(0)
            st.bar_chart(energy_counts)
            render_explanation(
                "Emotion intensity",
                "High voltage comments usually indicate strong opinions and strong sharing potential. Calm comments mean the topic is present, but not yet forcing a reaction.",
                "Turn high-voltage language into hooks, responses, and paid creative.",
            )

            selected_sentiment = st.selectbox(
                "Comment lane",
                ["All", "Positive", "Neutral", "Negative"],
                key="sentiment_lane",
            )
            if selected_sentiment == "All":
                filtered_comments = df.copy()
            else:
                filtered_comments = df[df["sentiment_label"] == selected_sentiment].copy()

            filtered_comments = filtered_comments.sort_values(
                "sentiment_score",
                ascending=selected_sentiment == "Negative",
            )
            st.markdown("<div class='section-label'>Comment explorer</div>", unsafe_allow_html=True)
            st.dataframe(
                filtered_comments[
                    [
                        "comment_text",
                        "platform",
                        "likes",
                        "sentiment_label",
                        "sentiment_score",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )
            render_explanation(
                "Comment explorer",
                "This table lets you inspect the raw comments behind the charts. It is useful for verifying whether a topic is genuinely important or just a statistical bump.",
                "Use the filter to isolate positive, neutral, or negative comments before writing replies.",
            )

        with detail_right:
            st.markdown("<div class='section-label'>Engagement by sentiment</div>", unsafe_allow_html=True)
            engagement = (
                df.groupby("sentiment_label")["likes"]
                .mean()
                .reindex(["Positive", "Neutral", "Negative"])
                .fillna(0)
            )
            st.bar_chart(engagement)
            render_explanation(
                "Engagement by sentiment",
                "This chart checks which tone gets the most likes. If positive comments outperform negative ones, that language is the best source for testimonials and campaign hooks.",
                "Reuse the highest-performing phrasing in your next message test.",
            )

            leaderboard = df.copy()
            max_likes = max(float(leaderboard["likes"].max()), 1.0)
            leaderboard["signal_strength"] = leaderboard["sentiment_score"].abs() + (
                leaderboard["likes"] / max_likes
            )
            leaderboard = leaderboard.sort_values("signal_strength", ascending=False).head(10)
            st.markdown("<div class='section-label'>Signal leaderboard</div>", unsafe_allow_html=True)
            st.dataframe(
                leaderboard[
                    [
                        "comment_text",
                        "platform",
                        "likes",
                        "sentiment_label",
                        "archetype",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )
            render_explanation(
                "Signal leaderboard",
                "These comments combine emotional weight and engagement. They are the best candidates for content ideas, support responses, or quote captures.",
                "Look here first when you want evidence-backed marketing copy.",
            )

    with topics_tab:
        if not topic_summaries:
            st.info(
                "Add more comments or loosen the filters a bit to unlock narrative clustering."
            )
        else:
            st.markdown(
                "<p class='helper-text'>The topic engine discovered {0} narrative lanes in this conversation.</p>".format(
                    actual_topics
                ),
                unsafe_allow_html=True,
            )
            for summary in topic_summaries:
                render_topic_card(summary)

    with language_tab:
        positive_keywords = keyword_frequency(
            df.loc[df["sentiment_label"] == "Positive", "comment_text"],
            limit=12,
        )
        negative_keywords = keyword_frequency(
            df.loc[df["sentiment_label"] == "Negative", "comment_text"],
            limit=12,
        )

        keyword_left, keyword_right = st.columns(2)
        with keyword_left:
            st.markdown("<div class='section-label'>Words driving affinity</div>", unsafe_allow_html=True)
            if positive_keywords.empty:
                st.info("No strong positive keyword pattern surfaced yet.")
            else:
                st.bar_chart(positive_keywords.set_index("keyword"))
                st.dataframe(positive_keywords, width="stretch", hide_index=True)
                render_explanation(
                    "Affirming language",
                    "These are the words customers use when they are leaning in. They are ideal raw material for headlines, testimonials, and social proof.",
                    "Mirror these phrases in campaign copy.",
                )

        with keyword_right:
            st.markdown("<div class='section-label'>Words driving friction</div>", unsafe_allow_html=True)
            if negative_keywords.empty:
                st.info("No strong negative keyword pattern surfaced yet.")
            else:
                st.bar_chart(negative_keywords.set_index("keyword"))
                st.dataframe(negative_keywords, width="stretch", hide_index=True)
                render_explanation(
                    "Friction language",
                    "These words often point to objections, doubts, or missing clarity. They are the fastest route to improving your messaging.",
                    "Build a response page or FAQ around the most repeated friction terms.",
                )

        question_examples = df[df["question_flag"]].head(6)
        st.markdown("<div class='section-label'>Questions the audience is already asking</div>", unsafe_allow_html=True)
        if question_examples.empty:
            st.markdown(
                """
                <div class="data-card">
                    <h3>Questions are quiet</h3>
                    <p>The audience is not asking much right now, which usually means the message is already clear or the creative is not yet sparking enough curiosity.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.dataframe(
                question_examples[["comment_text", "platform", "likes"]],
                width="stretch",
                hide_index=True,
            )
            render_explanation(
                "Question review",
                "Questions are a signal of curiosity and unresolved friction. They show what the audience still needs before it can convert or advocate.",
                "Convert the most repeated questions into a landing-page explainer or FAQ.",
            )

    with algorithm_tab:
        st.markdown(
            """
            <div class="section-label accent">Algorithm lab</div>
            <div class="section-hero" style="margin-top:0;">
                <div>
                    <h2>How the project works under the hood</h2>
                    <p>This section explains the main algorithms in plain language so the site stays transparent and easy to trust.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        algo_columns = st.columns(2, gap="large")
        with algo_columns[0]:
            render_algorithm_card(
                "TextBlob sentiment analysis",
                "Sentiment engine",
                "Each comment is scored for polarity so the app can separate positive, neutral, and negative reactions.",
                [
                    "Great for quick opinion mining",
                    "Creates the sentiment labels shown in the dashboard",
                    "Helps track message reception over time",
                ],
            )
            render_algorithm_card(
                "Heuristic audience tagging",
                "Pattern rules",
                "Simple keyword and punctuation rules detect questions, advocates, observers, and risk signals.",
                [
                    "Questions are caught with question marks",
                    "Strong positive/negative language is flagged",
                    "Useful when you want explainable labels",
                ],
            )

        with algo_columns[1]:
            render_algorithm_card(
                "TF-IDF + KMeans topic discovery",
                "Topic engine",
                "Comments are converted into TF-IDF vectors and clustered with KMeans to discover recurring narrative lanes.",
                [
                    "TF-IDF turns words into weighted features",
                    "KMeans groups similar comments together",
                    "Top terms are used to name each topic",
                ],
            )
            render_algorithm_card(
                "Trend and forecast logic",
                "Future insight layer",
                "The app looks at sentiment direction, question load, and topic balance to suggest the next marketing move.",
                [
                    "Creates future-facing insight cards",
                    "Highlights growth, risk, and opportunity",
                    "Turns chart readings into proposals",
                ],
            )

        st.markdown(
            """
            <div class="explain-card" style="margin-top:1.1rem;">
                <strong>End-to-end pipeline</strong>
                Raw comments -> cleaning -> sentiment scoring -> topic clustering -> explainable charts -> future insights and marketing proposals.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with strategy_tab:
        recommendations = build_recommendations(df, topic_summaries)
        recommendation_columns = st.columns(3)
        for column, (title, items) in zip(recommendation_columns, recommendations.items()):
            with column:
                render_recommendation_card(title, items)

        st.markdown("<div class='section-label accent'>Future insights</div>", unsafe_allow_html=True)
        future_insights = build_future_insights(df, topic_summaries, pulse)
        if future_insights:
            insight_columns = st.columns(min(3, len(future_insights)), gap="large")
            for column, insight in zip(insight_columns, future_insights):
                with column:
                    st.markdown(
                        """
                        <div class="forecast-card">
                            <h4>{title}</h4>
                            <div class="forecast-value">{value}</div>
                            <p>{note}</p>
                            <p style="margin-top:0.75rem;"><strong>Proposal:</strong> {proposal}</p>
                        </div>
                        """.format(
                            title=escape_html(insight["title"]),
                            value=escape_html(insight["value"]),
                            note=escape_html(insight["note"]),
                            proposal=escape_html(insight["proposal"]),
                        ),
                        unsafe_allow_html=True,
                    )

        strategy_left, strategy_right = st.columns([1.05, 0.95])
        with strategy_left:
            if topic_summaries and topic_summaries[0]["avg_score"] < 0:
                move_title = "Best next move: repair the dominant narrative"
                move_copy = "The biggest cluster is leaning negative, so the fastest win is a clarifying message that directly addresses the main complaint before it compounds into the next campaign cycle."
            elif topic_summaries:
                move_title = "Best next move: scale what is already resonating"
                move_copy = "Your largest topic cluster is already creating positive pull. Package that language into social proof, ad hooks, and follow-up content while the signal is warm."
            else:
                move_title = "Best next move: gather more signal"
                move_copy = "Sentiment is available, but topic intelligence is still light. Add more comments or reduce filtering so the strategy layer has richer material to work with."

            st.markdown("<div class='section-label'>Creative direction</div>", unsafe_allow_html=True)
            st.markdown(
                """
                <div class="story-card">
                    <h3>{title}</h3>
                    <p>{copy}</p>
                </div>
                """.format(
                    title=escape_html(move_title),
                    copy=escape_html(move_copy),
                ),
                unsafe_allow_html=True,
            )

        with strategy_right:
            export_columns = [
                "comment_text",
                "platform",
                "likes",
                "sentiment_label",
                "sentiment_score",
                "archetype",
            ]
            if "topic_cluster" in df.columns:
                export_columns.append("topic_cluster")

            st.markdown("<div class='section-label'>Shareable data slice</div>", unsafe_allow_html=True)
            st.dataframe(
                df[export_columns].head(12),
                width="stretch",
                hide_index=True,
            )


if "data_source" not in st.session_state:
    st.session_state["data_source"] = "Landing page"
if "youtube_data" not in st.session_state:
    st.session_state["youtube_data"] = None
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "Editorial Dawn"

st.sidebar.markdown("## Brand Intel Studio")
st.sidebar.markdown(
    """
    <div class="upload-note">
        <strong>Turn comment streams into brand direction.</strong><br><br>
        <div class="schema-row">
            <span class="schema-chip">Required: <code>comment_text</code></span>
            <span class="schema-chip">Best extra: <code>platform</code></span>
            <span class="schema-chip">Best extra: <code>likes</code></span>
            <span class="schema-chip">Best extra: <code>timestamp</code></span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("### Visual mode")
theme_mode = st.sidebar.selectbox(
    "Palette",
    THEME_OPTIONS,
    index=THEME_OPTIONS.index(st.session_state["theme_mode"])
    if st.session_state["theme_mode"] in THEME_OPTIONS
    else 0,
    help="Pick the visual mood that fits the story you want to tell.",
)
st.session_state["theme_mode"] = theme_mode

theme_caption = {
    "Editorial Dawn": "Bright, premium, and airy.",
    "Campaign Night": "Bold, cinematic, and high-contrast.",
    "Sunset Pulse": "Warm, modern, and brand-forward.",
}.get(theme_mode, "Bright, premium, and airy.")

theme_palette = {
    "Editorial Dawn": {
        "swatches": ["#ff6b35", "#138a72", "#f2a900"],
        "descriptor": "Clean editorial palette",
        "chips": ["Soft light", "Warm accent", "Fresh balance"],
    },
    "Campaign Night": {
        "swatches": ["#ff8a5b", "#25c79b", "#ffcb57"],
        "descriptor": "High-contrast campaign mode",
        "chips": ["Dark canvas", "Live signal", "Premium contrast"],
    },
    "Sunset Pulse": {
        "swatches": ["#ff7a4e", "#7e57c2", "#ffb703"],
        "descriptor": "Warm story-driven palette",
        "chips": ["Amber glow", "Modern warmth", "Brand-forward"],
    },
}.get(theme_mode, {
    "swatches": ["#ff6b35", "#138a72", "#f2a900"],
    "descriptor": "Clean editorial palette",
    "chips": ["Soft light", "Warm accent", "Fresh balance"],
})

st.sidebar.markdown(
    f"<div class='theme-toggle'><label>Theme mood</label><br><small>{escape_html(theme_caption)}</small></div>",
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
    <div class="visual-module">
        <label>Selected palette</label>
        <h4>{descriptor}</h4>
        <p>{caption}</p>
        <div class="theme-preview">
            <span class="theme-swatch" style="background:{s1};"></span>
            <span class="theme-swatch" style="background:{s2};"></span>
            <span class="theme-swatch" style="background:{s3};"></span>
        </div>
        <div class="theme-meta">
            <span>{chip1}</span>
            <span>{chip2}</span>
            <span>{chip3}</span>
        </div>
    </div>
    """.format(
        descriptor=escape_html(theme_palette["descriptor"]),
        caption=escape_html(theme_caption),
        s1=escape_html(theme_palette["swatches"][0]),
        s2=escape_html(theme_palette["swatches"][1]),
        s3=escape_html(theme_palette["swatches"][2]),
        chip1=escape_html(theme_palette["chips"][0]),
        chip2=escape_html(theme_palette["chips"][1]),
        chip3=escape_html(theme_palette["chips"][2]),
    ),
    unsafe_allow_html=True,
)

load_css(theme_mode)

data_source = st.sidebar.radio("Experience mode", SOURCE_OPTIONS, key="data_source")
source_df = None
source_label = "Campaign dataset"

if data_source == "Landing page":
    st.sidebar.info("Start with inspiration mode, or switch to sample, CSV, or YouTube when you want live analysis.")
    st.sidebar.button(
        "Open sample cockpit",
        use_container_width=True,
        on_click=set_data_source,
        args=("Use sample data",),
    )

elif data_source == "Use sample data":
    source_df = load_sample_data()
    source_label = "Sample campaign"
    st.sidebar.success("Demo dataset loaded.")

elif data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload a comment dataset",
        type=["csv"],
        help="A single CSV with a comment_text column is enough to start.",
    )
    source_label = "Uploaded dataset"
    if uploaded_file is not None:
        source_df = pd.read_csv(uploaded_file)
        st.sidebar.success("CSV loaded. Tune the controls below to shape the analysis.")

elif data_source == "Fetch from YouTube":
    source_label = "YouTube live sync"
    st.sidebar.markdown("### YouTube connector")

    default_key = ""
    try:
        if "YOUTUBE_API_KEY" in st.secrets and st.secrets["YOUTUBE_API_KEY"] != "REPLACE_WITH_YOUR_ACTUAL_API_KEY":
            default_key = st.secrets["YOUTUBE_API_KEY"]
            st.sidebar.success("API key loaded from Streamlit secrets.")
    except Exception:
        default_key = ""

    api_key = default_key
    if not api_key:
        api_key = st.sidebar.text_input(
            "YouTube API key",
            type="password",
            help="Add it to .streamlit/secrets.toml if you do not want to paste it each time.",
        )

    youtube_input = st.sidebar.text_input(
        "Video URL or video ID",
        placeholder="https://www.youtube.com/watch?v=...",
    )
    max_comments = st.sidebar.slider("Comments to fetch", 100, 500, 300, 50)

    if st.sidebar.button("Fetch comments", use_container_width=True):
        if not api_key:
            st.sidebar.error("Add a YouTube API key to fetch live comments.")
        elif not youtube_input:
            st.sidebar.error("Paste a YouTube URL or video ID first.")
        else:
            with st.spinner("Pulling comments from YouTube..."):
                youtube_df, error_message = fetch_youtube_comments(
                    api_key,
                    youtube_input,
                    max_comments=max_comments,
                )
            if error_message:
                st.sidebar.error(error_message)
                st.session_state["youtube_data"] = None
            else:
                st.session_state["youtube_data"] = youtube_df
                st.sidebar.success(
                    "Fetched {0} comments from YouTube.".format(len(youtube_df))
                )

    if st.session_state["youtube_data"] is not None:
        source_df = st.session_state["youtube_data"]

if source_df is not None:
    if "comment_text" not in source_df.columns:
        st.error("The dataset needs a comment_text column before I can build the experience.")
    else:
        st.sidebar.markdown("### Analysis controls")

        working_df = source_df.copy()
        if "platform" not in working_df.columns:
            working_df["platform"] = "Unknown"
        working_df["platform"] = working_df["platform"].fillna("Unknown").astype(str)

        available_platforms = sorted(
            [platform for platform in working_df["platform"].unique() if str(platform).strip()]
        )
        selected_platforms = st.sidebar.multiselect(
            "Platforms",
            options=available_platforms,
            default=available_platforms,
        )

        if "likes" not in working_df.columns:
            working_df["likes"] = 0
        working_df["likes"] = pd.to_numeric(working_df["likes"], errors="coerce").fillna(0)
        max_likes = int(working_df["likes"].max()) if not working_df.empty else 0
        min_likes = st.sidebar.slider(
            "Minimum likes",
            min_value=0,
            max_value=max(0, max_likes),
            value=0,
        )

        keyword_filter = st.sidebar.text_input(
            "Only include comments matching",
            placeholder="delivery, pricing, support",
        )
        requested_topics = st.sidebar.slider(
            "Narrative clusters",
            min_value=2,
            max_value=8,
            value=4,
        )

        if selected_platforms:
            working_df = working_df[working_df["platform"].isin(selected_platforms)]
        else:
            working_df = working_df.iloc[0:0]

        working_df = working_df[working_df["likes"] >= min_likes]

        if keyword_filter:
            working_df = working_df[
                working_df["comment_text"].fillna("").astype(str).str.contains(
                    keyword_filter,
                    case=False,
                    na=False,
                    regex=False,
                )
            ]

        if working_df.empty:
            st.warning(
                "Those filters removed every comment. Try widening the platform, like, or keyword filters and we will bring the cockpit back to life."
            )
        else:
            with st.spinner("Building the command center..."):
                processed_df, topic_keywords, actual_topics = process_comments(
                    working_df,
                    requested_topics,
                )
            render_dashboard(processed_df, topic_keywords, actual_topics, source_label)
else:
    render_landing_page(data_source)
