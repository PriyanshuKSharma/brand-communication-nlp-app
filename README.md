# 🧠 Brand Intel: AI-Powered Social Media Analytics

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://github.com/PriyanshuKSharma/brand-communication-nlp-app)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-t he-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NLP](https://img.shields.io/badge/AI-Natural%20Language%20Processing-purple?style=for-the-badge)](https://en.wikipedia.org/wiki/Natural_language_processing)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> **Turn messy social media comments into actionable brand strategy.**

---

## � Overview

**Brand Intel** is a cutting-edge analytics dashboard designed for marketing teams and brand managers. It leverages **Natural Language Processing (NLP)** to "listen" to social media conversations at scale.

Instead of manually reading thousands of comments, this tool automatically:

1.  **Fetches** real-time data from YouTube campaigns.
2.  **Decodes** human sentiment (Positive vs. Negative).
3.  **Clusters** discussions into key topics (e.g., "Battery Life", "Pricing", "Customer Service").
4.  **Recommends** strategic actions to improve brand perception.

---

## ✨ Key Features

| Feature                    | Description                                                                       |
| :------------------------- | :-------------------------------------------------------------------------------- |
| **📺 YouTube Integration** | Fetch comments directly from any YouTube video URL via API.                       |
| **❤️ Sentiment Engine**    | Classifies feedback as **Positive**, **Neutral**, or **Negative** with precision. |
| **🔍 Topic Modeling**      | Uses **K-Means Clustering** to discover hidden themes in user discussions.        |
| **📊 Interactive Viz**     | Beautiful, interactive charts and data tables powered by Streamlit.               |
| **💡 Smart Strategy**      | Auto-generates actionable recommendations based on data patterns.                 |
| **🌑 Dark Mode**           | Sleek, professional dark-themed UI for focused analysis.                          |

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) (Python-based web framework)
- **Core Language**: Python 3.x
- **NLP & ML**:
  - `TextBlob` (Sentiment Analysis)
  - `Scikit-learn` (TF-IDF Vectorization, K-Means Clustering)
- **Data Handling**: `Pandas`, `NumPy`
- **API**: Google YouTube Data API v3

---

## ⚡ Quick Start Guide

### 1. Clone the Repository

```bash
git clone https://github.com/PriyanshuKSharma/brand-communication-nlp-app.git
cd brand-communication-nlp-app
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download NLP Corpora

Required for the sentiment analysis engine:

```bash
python -m textblob.download_corpora
```

### 4. Set Up API Key (Optional but Recommended)

To fetch real YouTube data, you need a Google API Key.

1.  Create a file `.streamlit/secrets.toml`.
2.  Add your key:
    ```toml
    YOUTUBE_API_KEY = "YOUR_GOOGLE_API_KEY"
    ```

### 5. Launch the Dashboard

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```text
brand-communication-nlp-app/
├── 📂 data/                  # Sample datasets for testing
├── 📂 .streamlit/            # Configuration & Secrets
│   ├── config.toml           # UI Theme settings
│   └── secrets.toml          # API Keys (Gitignored)
├── app.py                    # 🚀 Main Application Entry Point
├── nlp_utils.py              # 🧠 NLP Helper Functions (Cleaning, Modeling)
├── requirements.txt          # 📦 Project Dependencies
└── README.md                 # 📄 Documentation
```

---

## 🎯 Case Study Example: Samsung S24 Ultra

This project includes a case study analyzing the **Samsung Galaxy S24 Ultra** launch campaign.

- **Objective**: Understand user reaction to the new AI features and pricing.
- **Findings**:
  - ✅ **Positive**: Users love the "Galaxy AI" features and camera zoom.
  - ❌ **Negative**: Significant complaints about the price hike and trade-in values.
- **Action**: The tool recommended emphasizing trade-in deals in follow-up FAQs to mitigate price concerns.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---

## �📬 Contact

**Priyanshu Sharma**

- GitHub: [@PriyanshuKSharma](https://github.com/PriyanshuKSharma)

---

## _Built with ❤️ and ☕ using Python._

## 📚 Detailed Case Study Report

### 1. Executive Summary

In the digital age, brands receive thousands of comments across platforms like Instagram, YouTube, and X (Twitter). However, manually analyzing this feedback is impossible at scale. This project presents a **Streamlit-based NLP Dashboard** that automates the analysis of user comments. By extracting sentiment trends and identifying recurring topics, the tool provides actionable insights to refine brand messaging, improve customer support, and optimize ad campaigns.

### 2. Problem Statement

Marketing and communication teams often face the following challenges:

- **Data Overload**: Too many comments to read manually.
- **Missed Insights**: "Gut feeling" is used instead of data to judge campaign reception.
- **Unaddressed Pain Points**: Recurring complaints (e.g., "delivery delay", "high price") get lost in the noise.
- **Tone Deafness**: Brands may continue using a tone that doesn't resonate with their audience.

**The Solution**: An automated pipeline that ingests comments, processes them using Natural Language Processing (NLP), and outputs a strategic dashboard.

### 3. Objectives

The primary goals of this tool are:

1.  **Quantify Public Sentiment**: Determine if the overall reaction to a post or campaign is Positive, Neutral, or Negative.
2.  **Identify Key Topics**: Automatically cluster comments into themes (e.g., "Product Quality", "Shipping Issues", "Customer Service") without manual tagging.
3.  **Generate Actionable Recommendations**: Provide data-backed suggestions for FAQ updates, content adjustments, and crisis management.

### 4. Technical Methodology (The NLP Pipeline)

The application follows a standard NLP pipeline to transform raw text into insights:

#### Step 1: Data Ingestion

- **Input**: A CSV file containing social media comments.
- **Required Fields**: `comment_text`.
- **Optional Fields**: `platform`, `timestamp`, `likes`.

#### Step 2: Text Pre-processing

Before analysis, raw text is cleaned to ensure accuracy:

- **Lowercasing**: "Great" and "great" are treated the same.
- **Noise Removal**: URLs, emojis, and special characters are removed.
- **Handle/Hashtag Removal**: `@mentions` and `#hashtags` are stripped to focus on the message content.
- **Stopword Removal**: Common words like "the", "is", "and" are removed (during vectorization).

#### Step 3: Sentiment Analysis

- **Library**: `TextBlob`
- **Method**: Rule-based polarity scoring.
- **Output**:
  - **Score**: Float value between -1.0 (Negative) and +1.0 (Positive).
  - **Label**: Classified as Positive (> 0.1), Negative (< -0.1), or Neutral.

#### Step 4: Topic Modeling (Unsupervised Learning)

- **Vectorization**: `TF-IDF` (Term Frequency-Inverse Document Frequency) converts text into numerical vectors, highlighting unique/important words.
- **Clustering**: `K-Means Clustering` groups similar comments together based on their vocabulary.
- **Output**: Distinct "topics" (clusters) represented by their top keywords.

### 5. Application Walkthrough

The **Streamlit** web application serves as the interface for these insights.

#### Sidebar: Controls

- **Upload Data**: User uploads their specific dataset.
- **Use Sample Data**: A toggle to load a synthetic dataset for demonstration.
- **Platform Filter**: Filter analysis by source (e.g., only Instagram comments).
- **Topic Slider**: Adjust the number of clusters (K) for the K-Means algorithm.

#### Main Dashboard Tabs

1.  **Overview**: High-level metrics (total comments, sentiment distribution chart).
2.  **Sentiment Analysis**: Deep dive into positive vs. negative feedback. Includes a data explorer to read specific comments.
3.  **Topic Explorer**: Displays the discovered topics and their top keywords. Users can expand sections to read sample comments from each topic.
4.  **Keyword Insights**: A frequency analysis showing the most common words used by customers.
5.  **Recommendations**: The "Strategy" section. It heuristically analyzes the data to suggest:
    - **Pain Points**: Negative topics that need addressing in FAQs.
    - **Winning Themes**: Positive topics to emphasize in future marketing.

### 6. Future Improvements

- **Advanced Models**: Replace TF-IDF/K-Means with BERTopic or LDA for better topic coherence.
- **Deep Learning Sentiment**: Use RoBERTa or DistilBERT for more nuanced sentiment detection (sarcasm, context).
- **Real-time API**: Connect directly to YouTube/Instagram APIs to fetch live comments.
- **Named Entity Recognition (NER)**: Automatically detect product names or competitor mentions.
