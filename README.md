# Brand Intel Studio

Brand Intel Studio turns raw comment streams into a polished brand intelligence experience.

Instead of a basic dashboard, the app now opens with a real landing page and then moves into a multi-panel command center for:

- sentiment decoding
- narrative clustering
- audience question detection
- keyword friction and affinity analysis
- strategy recommendations for what to repair, amplify, and monitor next

## What is inside

- `Landing page` for the product story and a one-click sample demo
- `Mission Control` for the overall conversation mix and timeline pulse
- `Signal Deep Dive` for positive and negative language review
- `Topic Radar` for clustered narrative lanes
- `Language Lab` for high-frequency positive and negative keywords
- `Strategy Studio` for communication guidance and exportable enriched data

## Data sources

You can run the experience with:

- the included sample dataset
- your own CSV upload
- YouTube comments fetched from a video URL or video ID

Minimum CSV requirement:

```text
comment_text
```

Helpful extra columns:

```text
platform, likes, timestamp, published_at
```

## Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Download TextBlob corpora:

```bash
python -m textblob.download_corpora
```

3. Optional: add a YouTube API key to `.streamlit/secrets.toml`

```toml
YOUTUBE_API_KEY = "YOUR_API_KEY"
```

4. Launch the app:

```bash
streamlit run app.py
```

## Project structure

```text
brand-communication-nlp-app/
├── app.py
├── nlp_utils.py
├── requirements.txt
├── data/
│   └── sample_comments.csv
└── report/
```

## Core stack

- Streamlit
- Pandas
- NumPy
- TextBlob
- scikit-learn
- YouTube Data API v3

## Next ideas

- add richer topic modeling with BERTopic or transformer embeddings
- track sentiment movement across multiple campaigns over time
- generate suggested reply copy for each negative narrative cluster
