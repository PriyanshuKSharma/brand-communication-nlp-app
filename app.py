import html
import os
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
THEME_OPTIONS = ("Editorial", "Campaign Night")


def escape_html(value):
    return html.escape(str(value))


def load_css(theme_mode):
    night_mode = theme_mode == "Campaign Night"
    app_bg = (
        """
                radial-gradient(circle at top left, rgba(255, 107, 53, 0.20), transparent 28%),
                radial-gradient(circle at 86% 8%, rgba(19, 138, 114, 0.22), transparent 24%),
                radial-gradient(circle at 20% 82%, rgba(242, 169, 0, 0.16), transparent 20%),
                linear-gradient(180deg, #f7f2e8 0%, #efe7d8 100%);
        """
        if not night_mode
        else
        """
                radial-gradient(circle at top left, rgba(255, 107, 53, 0.28), transparent 28%),
                radial-gradient(circle at 85% 10%, rgba(24, 197, 160, 0.22), transparent 24%),
                radial-gradient(circle at 18% 82%, rgba(255, 200, 87, 0.16), transparent 20%),
                linear-gradient(180deg, #08111f 0%, #101b2f 100%);
        """
    )
    card_bg = "rgba(255, 252, 247, 0.82)" if not night_mode else "rgba(13, 22, 38, 0.82)"
    card_text = "#11203b" if not night_mode else "#f5f7fb"
    muted = "#5c6270" if not night_mode else "rgba(245, 247, 251, 0.72)"
    app_ink = "#11203b" if not night_mode else "#f5f7fb"
    sidebar_bg = (
        "linear-gradient(180deg, rgba(11, 19, 41, 0.98) 0%, rgba(22, 32, 61, 0.96) 100%)"
        if not night_mode
        else "linear-gradient(180deg, rgba(5, 10, 19, 0.98) 0%, rgba(11, 18, 31, 0.96) 100%)"
    )
    board_bg = (
        "linear-gradient(180deg, rgba(17, 32, 59, 0.96) 0%, rgba(25, 37, 70, 0.95) 100%)"
        if not night_mode
        else "linear-gradient(180deg, rgba(9, 16, 28, 0.98) 0%, rgba(20, 30, 50, 0.96) 100%)"
    )
    border_soft = "rgba(17, 32, 59, 0.08)" if not night_mode else "rgba(255, 255, 255, 0.10)"
    panel_shadow = "0 22px 60px rgba(17, 32, 59, 0.12)" if not night_mode else "0 22px 60px rgba(0, 0, 0, 0.28)"
    card_panel = card_bg
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Space+Grotesk:wght@400;500;700&display=swap');

        :root {
            --bg: #f4efe5;
            --ink: %s;
            --panel: %s;
            --muted: %s;
            --accent: #ff6b35;
            --accent-2: #138a72;
            --gold: #f2a900;
            --line: %s;
            --shadow: %s;
        }

        .stApp {
            background: %s;
            color: %s;
        }

        [data-testid="stAppViewContainer"] {
            background: transparent;
        }

        .block-container {
            max-width: 1220px;
            padding-top: 1.5rem;
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
            background: %s;
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
            margin-bottom: 1rem;
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
            background: rgba(255, 252, 247, 0.78);
            border: 1px solid rgba(17, 32, 59, 0.08);
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
            padding: 2rem;
            border-radius: 30px;
            background: %s;
            border: 1px solid %s;
            box-shadow: var(--shadow);
            overflow: hidden;
            position: relative;
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
            gap: 1.2rem;
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
            margin-bottom: 1rem;
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
            margin-top: 1.25rem;
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
            padding: 1.15rem;
            background: %s;
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
            margin-top: 1.75rem;
            margin-bottom: 0.8rem;
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--muted);
        }

        .feature-card,
        .story-card,
        .impact-card,
        .topic-card,
        .quote-card,
        .reco-card,
        .data-card {
            height: 100%;
            background: %s;
            border: 1px solid %s;
            border-radius: 24px;
            padding: 1.25rem;
            box-shadow: var(--shadow);
        }

        .feature-card h3,
        .story-card h3,
        .impact-card h3,
        .topic-card h3,
        .quote-card h3,
        .reco-card h3,
        .data-card h3 {
            margin-top: 0;
            margin-bottom: 0.55rem;
            font-size: 1.2rem;
        }

        .feature-card p,
        .story-card p,
        .impact-card p,
        .topic-card p,
        .quote-card p,
        .reco-card p,
        .data-card p {
            color: var(--muted);
            line-height: 1.65;
            margin-bottom: 0;
        }

        .feature-card {
            position: relative;
            overflow: hidden;
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

        .feature-icon {
            width: 44px;
            height: 44px;
            border-radius: 14px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: rgba(17, 32, 59, 0.06);
            margin-bottom: 0.9rem;
            font-size: 1.2rem;
        }

        .metric-card {
            border-radius: 24px;
            padding: 1.15rem;
            background: rgba(255, 252, 247, 0.82);
            border: 1px solid rgba(17, 32, 59, 0.08);
            box-shadow: var(--shadow);
            min-height: 152px;
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

        .topic-chip {
            display: inline-flex;
            margin-right: 0.45rem;
            margin-bottom: 0.45rem;
            padding: 0.35rem 0.65rem;
            border-radius: 999px;
            background: rgba(17, 32, 59, 0.06);
            font-size: 0.84rem;
            font-weight: 600;
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

        .mini-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.9rem;
            margin-top: 1rem;
        }

        .mini-stat {
            border-radius: 20px;
            padding: 0.95rem 1rem;
            background: %s;
            border: 1px solid %s;
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
            margin: 1.6rem 0 0.8rem;
            display: flex;
            align-items: end;
            justify-content: space-between;
            gap: 1rem;
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
            padding: 1rem 1.1rem;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.12);
            color: #f7f4ee;
            line-height: 1.55;
        }

        .theme-toggle {
            margin-top: 0.75rem;
            padding: 0.95rem 1rem;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.12);
        }

        .theme-toggle label {
            font-weight: 700;
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
            .hero-shell {
                padding: 1.4rem;
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
        """ % (
            app_ink,
            card_panel,
            muted,
            border_soft,
            panel_shadow,
            app_bg,
            app_ink,
            sidebar_bg,
            "linear-gradient(135deg, rgba(255, 248, 240, 0.95) 0%, rgba(255, 255, 255, 0.74) 52%, rgba(226, 246, 241, 0.9) 100%)"
            if not night_mode
            else "linear-gradient(135deg, rgba(13, 22, 38, 0.96) 0%, rgba(20, 30, 50, 0.92) 52%, rgba(12, 62, 53, 0.82) 100%)",
            border_soft,
            board_bg,
            card_panel,
            border_soft,
            "rgba(255, 255, 255, 0.66)" if not night_mode else "rgba(255, 255, 255, 0.08)",
            border_soft,
        ),
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
        dominant_value = ", ".join(dominant_topic["keywords"].split(", ")[:2]).title()
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
        lead_topic = ", ".join(lead["keywords"].split(", ")[:3]).title()
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
        win_topic = ", ".join(win["keywords"].split(", ")[:3]).title()
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
        dominant = ", ".join(topic_summaries[0]["keywords"].split(", ")[:3]).title()
        recommendations["Watch next"].append(
            "Your largest conversation cluster is {0}. Track whether it stays opportunity-led or slides into friction over time.".format(
                dominant
            )
        )

    return recommendations


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
    keyword_markup = "".join(
        "<span class='topic-chip'>{0}</span>".format(escape_html(keyword.strip().title()))
        for keyword in summary["keywords"].split(",")[:5]
        if keyword.strip()
    )

    example_markup = "".join(
        "<p style='margin-top:0.7rem;'><strong>Signal:</strong> {0}</p>".format(
            escape_html(example)
        )
        for example in summary["examples"]
    )

    st.markdown(
        """
        <div class="topic-card">
            <h3>Topic {topic_number}</h3>
            <p><span class="{tone_class}">{tone_text}</span> - {count} comments, {share}% of the conversation.</p>
            <div style="margin-top:0.8rem; margin-bottom:0.4rem;">{keyword_markup}</div>
            {example_markup}
        </div>
        """.format(
            topic_number=summary["id"] + 1,
            tone_class=summary["tone_class"],
            tone_text=escape_html(summary["tone_text"]),
            count=summary["count"],
            share=int(round(summary["share"] * 100)),
            keyword_markup=keyword_markup,
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
        "Upload CSV": "Your cockpit is ready. Add a CSV in the sidebar and the narrative engine will spin up instantly.",
        "Fetch from YouTube": "Paste a video URL and API key in the sidebar to turn live comment streams into strategy.",
        "Use sample data": "The demo dataset is one click away if you want to experience the full product flow first.",
    }.get(mode_hint, "Load a dataset from the sidebar to start the analysis.")

    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-grid">
                <div>
                    <div class="eyebrow eyebrow-variant">Brand Intel Studio</div>
                    <div class="hero-title">Turn social noise into a sharp brand signal.</div>
                    <p class="hero-copy">This project is built as a modern comment intelligence studio for brands, creators, and marketers. It blends sentiment decoding, topic discovery, and strategy suggestions into one polished experience that feels more like a product launch than a school dashboard.</p>
                    <p class="hero-copy">{helper_copy}</p>
                    <div class="hero-actions">
                        <span class="signal-pill accent">Start with the demo</span>
                        <span class="signal-pill teal">Upload your own comments</span>
                        <span class="signal-pill">Connect YouTube live</span>
                    </div>
                    <div class="hero-badge-row">
                        <span class="signal-pill">{sample_comments} demo comments</span>
                        <span class="signal-pill">{sample_platforms} channels represented</span>
                        <span class="signal-pill">{sample_questions} audience questions</span>
                    </div>
                </div>
                <div class="signal-board">
                    <div class="eyebrow" style="background: rgba(255,255,255,0.08); color: #fff8f1;">Live signal board</div>
                    <h3>What the demo already knows</h3>
                    <p>Use the sample dataset to explore an immediate, visual read on how a brand is being perceived before you bring in live data.</p>
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

    action_col, note_col = st.columns([0.28, 0.72])
    with action_col:
        st.button(
            "Launch sample cockpit",
            use_container_width=True,
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
        <div class="section-hero">
            <div>
                <h2>What makes it feel different</h2>
                <p>A stronger landing page should make the idea obvious in seconds: this is a studio for turning public feedback into decisions, not just a chart wall.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    feature_columns = st.columns(3)
    feature_cards = [
        (
            "Designed like a product",
            "The first screen now tells a clear story, gives the user a confident next step, and uses a more premium editorial layout.",
            "01",
            "Hero experience",
        ),
        (
            "Built for curiosity",
            "The landing page shows what the app does before asking for input, which makes the project feel more inventive and approachable.",
            "02",
            "Story first",
        ),
        (
            "Ready for real work",
            "A live demo path, upload path, and YouTube path are all visible from the start so the user never feels stuck.",
            "03",
            "Multiple entry points",
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

    story_left, story_right = st.columns([1.15, 0.85])
    with story_left:
        st.markdown(
            """
            <div class="story-card">
                <div class="feature-kicker">How it works</div>
                <h3>From noisy comments to sharper messaging</h3>
                <p><strong>1. Ingest.</strong> Bring in a CSV, load the demo, or connect a YouTube campaign.</p>
                <p style="margin-top:0.65rem;"><strong>2. Decode.</strong> The app scores sentiment, detects audience questions, and groups the conversation into narrative lanes.</p>
                <p style="margin-top:0.65rem;"><strong>3. Direct.</strong> Strategy cards tell the team what to repair, what to amplify, and what to monitor next.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with story_right:
        st.markdown(
            """
            <div class="data-card">
                <div class="feature-kicker">Input shape</div>
                <h3>What the app expects</h3>
                <p><strong>Required:</strong> `comment_text`</p>
                <p style="margin-top:0.55rem;"><strong>Helpful:</strong> `platform`, `likes`, `timestamp`, `published_at`</p>
                <p style="margin-top:0.55rem;">If those extra columns are missing, the app still works and fills in smart defaults.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.code("comment_text, platform, likes, timestamp", language="text")


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
            use_container_width=True,
        )
    with action_col_2:
        st.button(
            "Return to landing",
            use_container_width=True,
            on_click=set_data_source,
            args=("Landing page",),
        )
    with action_col_3:
        st.markdown(
            "<p class='helper-text'>Use the sidebar to tune platforms, likes, keywords, and the number of narrative clusters.</p>",
            unsafe_allow_html=True,
        )

    render_metric_cards(metrics)

    mission_tab, signal_tab, topics_tab, language_tab, strategy_tab = st.tabs(
        [
            "Mission Control",
            "Signal Deep Dive",
            "Topic Radar",
            "Language Lab",
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

            st.markdown("<div class='section-label'>Platform spread</div>", unsafe_allow_html=True)
            platform_counts = df["platform"].fillna("Unknown").value_counts()
            st.bar_chart(platform_counts)

            top_rows = (
                df.sort_values(["likes", "sentiment_score"], ascending=[False, False])
                .loc[:, ["comment_text", "platform", "likes", "sentiment_label"]]
                .head(6)
            )
            st.markdown("<div class='section-label'>High-signal comments</div>", unsafe_allow_html=True)
            st.dataframe(top_rows, use_container_width=True, hide_index=True)

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
                use_container_width=True,
                hide_index=True,
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
                use_container_width=True,
                hide_index=True,
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
            for start in range(0, len(topic_summaries), 2):
                row_items = topic_summaries[start : start + 2]
                row_columns = st.columns(len(row_items))
                for column, summary in zip(row_columns, row_items):
                    with column:
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
                st.dataframe(positive_keywords, use_container_width=True, hide_index=True)

        with keyword_right:
            st.markdown("<div class='section-label'>Words driving friction</div>", unsafe_allow_html=True)
            if negative_keywords.empty:
                st.info("No strong negative keyword pattern surfaced yet.")
            else:
                st.bar_chart(negative_keywords.set_index("keyword"))
                st.dataframe(negative_keywords, use_container_width=True, hide_index=True)

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
                use_container_width=True,
                hide_index=True,
            )

    with strategy_tab:
        recommendations = build_recommendations(df, topic_summaries)
        recommendation_columns = st.columns(3)
        for column, (title, items) in zip(recommendation_columns, recommendations.items()):
            with column:
                render_recommendation_card(title, items)

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
                use_container_width=True,
                hide_index=True,
            )


load_css()

if "data_source" not in st.session_state:
    st.session_state["data_source"] = "Landing page"
if "youtube_data" not in st.session_state:
    st.session_state["youtube_data"] = None

st.sidebar.markdown("## Brand Intel Studio")
st.sidebar.markdown(
    """
    <div class="upload-note">
        <strong>Turn comment streams into brand direction.</strong><br><br>
        Required column: <code>comment_text</code><br>
        Best extra fields: <code>platform</code>, <code>likes</code>, <code>timestamp</code>
    </div>
    """,
    unsafe_allow_html=True,
)

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
    if "YOUTUBE_API_KEY" in st.secrets and st.secrets["YOUTUBE_API_KEY"] != "REPLACE_WITH_YOUR_ACTUAL_API_KEY":
        default_key = st.secrets["YOUTUBE_API_KEY"]
        st.sidebar.success("API key loaded from Streamlit secrets.")

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
