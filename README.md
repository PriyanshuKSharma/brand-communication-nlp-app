# 📱 Social Media Analytics & Brand Strategy Case Study

## **Project Title**

**Using NLP on Social Media Comments to Improve Brand Communication Strategy**

---

## **1. Executive Summary**

In the digital age, brands receive thousands of comments across platforms like Instagram, YouTube, and X (Twitter). However, manually analyzing this feedback is impossible at scale. This project presents a **Streamlit-based NLP Dashboard** that automates the analysis of user comments. By extracting sentiment trends and identifying recurring topics, the tool provides actionable insights to refine brand messaging, improve customer support, and optimize ad campaigns.

---

## **2. Problem Statement**

Marketing and communication teams often face the following challenges:

- **Data Overload**: Too many comments to read manually.
- **Missed Insights**: "Gut feeling" is used instead of data to judge campaign reception.
- **Unaddressed Pain Points**: Recurring complaints (e.g., "delivery delay", "high price") get lost in the noise.
- **Tone Deafness**: Brands may continue using a tone that doesn't resonate with their audience.

**The Solution**: An automated pipeline that ingests comments, processes them using Natural Language Processing (NLP), and outputs a strategic dashboard.

---

## **3. Objectives**

The primary goals of this tool are:

1.  **Quantify Public Sentiment**: Determine if the overall reaction to a post or campaign is Positive, Neutral, or Negative.
2.  **Identify Key Topics**: Automatically cluster comments into themes (e.g., "Product Quality", "Shipping Issues", "Customer Service") without manual tagging.
3.  **Generate Actionable Recommendations**: Provide data-backed suggestions for FAQ updates, content adjustments, and crisis management.

---

## **4. Technical Methodology (The NLP Pipeline)**

The application follows a standard NLP pipeline to transform raw text into insights:

### **Step 1: Data Ingestion**

- **Input**: A CSV file containing social media comments.
- **Required Fields**: `comment_text`.
- **Optional Fields**: `platform`, `timestamp`, `likes`.

### **Step 2: Text Pre-processing**

Before analysis, raw text is cleaned to ensure accuracy:

- **Lowercasing**: "Great" and "great" are treated the same.
- **Noise Removal**: URLs, emojis, and special characters are removed.
- **Handle/Hashtag Removal**: `@mentions` and `#hashtags` are stripped to focus on the message content.
- **Stopword Removal**: Common words like "the", "is", "and" are removed (during vectorization).

### **Step 3: Sentiment Analysis**

- **Library**: `TextBlob`
- **Method**: Rule-based polarity scoring.
- **Output**:
  - **Score**: Float value between -1.0 (Negative) and +1.0 (Positive).
  - **Label**: Classified as Positive (> 0.1), Negative (< -0.1), or Neutral.

### **Step 4: Topic Modeling (Unsupervised Learning)**

- **Vectorization**: `TF-IDF` (Term Frequency-Inverse Document Frequency) converts text into numerical vectors, highlighting unique/important words.
- **Clustering**: `K-Means Clustering` groups similar comments together based on their vocabulary.
- **Output**: Distinct "topics" (clusters) represented by their top keywords.

---

## **5. Application Walkthrough**

The **Streamlit** web application serves as the interface for these insights.

### **Sidebar: Controls**

- **Upload Data**: User uploads their specific dataset.
- **Use Sample Data**: A toggle to load a synthetic dataset for demonstration.
- **Platform Filter**: Filter analysis by source (e.g., only Instagram comments).
- **Topic Slider**: Adjust the number of clusters (K) for the K-Means algorithm.

### **Main Dashboard Tabs**

1.  **Overview**: High-level metrics (total comments, sentiment distribution chart).
2.  **Sentiment Analysis**: Deep dive into positive vs. negative feedback. Includes a data explorer to read specific comments.
3.  **Topic Explorer**: Displays the discovered topics and their top keywords. Users can expand sections to read sample comments from each topic.
4.  **Keyword Insights**: A frequency analysis showing the most common words used by customers.
5.  **Recommendations**: The "Strategy" section. It heuristically analyzes the data to suggest:
    - **Pain Points**: Negative topics that need addressing in FAQs.
    - **Winning Themes**: Positive topics to emphasize in future marketing.

---

## **6. Project Structure**

```text
brand-nlp-case-study/
├── app.py                  # The main Streamlit application entry point
├── nlp_utils.py            # Helper module containing cleaning, sentiment, and clustering functions
├── requirements.txt        # List of Python dependencies
├── README.md               # This project documentation
└── data/
    └── sample_comments.csv # Synthetic dataset for testing and demonstration
```

# 📱 Social Media Analytics & Brand Strategy Case Study

## **Project Title**

**Using NLP on Social Media Comments to Improve Brand Communication Strategy**

---

## **1. Executive Summary**

In the digital age, brands receive thousands of comments across platforms like Instagram, YouTube, and X (Twitter). However, manually analyzing this feedback is impossible at scale. This project presents a **Streamlit-based NLP Dashboard** that automates the analysis of user comments. By extracting sentiment trends and identifying recurring topics, the tool provides actionable insights to refine brand messaging, improve customer support, and optimize ad campaigns.

---

## **2. Problem Statement**

Marketing and communication teams often face the following challenges:

- **Data Overload**: Too many comments to read manually.
- **Missed Insights**: "Gut feeling" is used instead of data to judge campaign reception.
- **Unaddressed Pain Points**: Recurring complaints (e.g., "delivery delay", "high price") get lost in the noise.
- **Tone Deafness**: Brands may continue using a tone that doesn't resonate with their audience.

**The Solution**: An automated pipeline that ingests comments, processes them using Natural Language Processing (NLP), and outputs a strategic dashboard.

---

## **3. Objectives**

The primary goals of this tool are:

1.  **Quantify Public Sentiment**: Determine if the overall reaction to a post or campaign is Positive, Neutral, or Negative.
2.  **Identify Key Topics**: Automatically cluster comments into themes (e.g., "Product Quality", "Shipping Issues", "Customer Service") without manual tagging.
3.  **Generate Actionable Recommendations**: Provide data-backed suggestions for FAQ updates, content adjustments, and crisis management.

---

## **4. Technical Methodology (The NLP Pipeline)**

The application follows a standard NLP pipeline to transform raw text into insights:

### **Step 1: Data Ingestion**

- **Input**: A CSV file containing social media comments.
- **Required Fields**: `comment_text`.
- **Optional Fields**: `platform`, `timestamp`, `likes`.

### **Step 2: Text Pre-processing**

Before analysis, raw text is cleaned to ensure accuracy:

- **Lowercasing**: "Great" and "great" are treated the same.
- **Noise Removal**: URLs, emojis, and special characters are removed.
- **Handle/Hashtag Removal**: `@mentions` and `#hashtags` are stripped to focus on the message content.
- **Stopword Removal**: Common words like "the", "is", "and" are removed (during vectorization).

### **Step 3: Sentiment Analysis**

- **Library**: `TextBlob`
- **Method**: Rule-based polarity scoring.
- **Output**:
  - **Score**: Float value between -1.0 (Negative) and +1.0 (Positive).
  - **Label**: Classified as Positive (> 0.1), Negative (< -0.1), or Neutral.

### **Step 4: Topic Modeling (Unsupervised Learning)**

- **Vectorization**: `TF-IDF` (Term Frequency-Inverse Document Frequency) converts text into numerical vectors, highlighting unique/important words.
- **Clustering**: `K-Means Clustering` groups similar comments together based on their vocabulary.
- **Output**: Distinct "topics" (clusters) represented by their top keywords.

---

## **5. Application Walkthrough**

The **Streamlit** web application serves as the interface for these insights.

### **Sidebar: Controls**

- **Upload Data**: User uploads their specific dataset.
- **Use Sample Data**: A toggle to load a synthetic dataset for demonstration.
- **Platform Filter**: Filter analysis by source (e.g., only Instagram comments).
- **Topic Slider**: Adjust the number of clusters (K) for the K-Means algorithm.

### **Main Dashboard Tabs**

1.  **Overview**: High-level metrics (total comments, sentiment distribution chart).
2.  **Sentiment Analysis**: Deep dive into positive vs. negative feedback. Includes a data explorer to read specific comments.
3.  **Topic Explorer**: Displays the discovered topics and their top keywords. Users can expand sections to read sample comments from each topic.
4.  **Keyword Insights**: A frequency analysis showing the most common words used by customers.
5.  **Recommendations**: The "Strategy" section. It heuristically analyzes the data to suggest:
    - **Pain Points**: Negative topics that need addressing in FAQs.
    - **Winning Themes**: Positive topics to emphasize in future marketing.

---

## **6. Project Structure**

```text
brand-nlp-case-study/
├── app.py                  # The main Streamlit application entry point
├── nlp_utils.py            # Helper module containing cleaning, sentiment, and clustering functions
├── requirements.txt        # List of Python dependencies
├── README.md               # This project documentation
└── data/
    └── sample_comments.csv # Synthetic dataset for testing and demonstration
```

---

## **7. Installation & Usage**

### **Prerequisites**

- Python 3.8 or higher
- pip (Python package manager)

### **Setup Instructions**

1.  **Clone/Download** the project folder.
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Download NLP Corpora** (for TextBlob):
    ```bash
    python -m textblob.download_corpora
    ```
4.  **Run the Application**:
    ```bash
    streamlit run app.py
    ```
5.  **Access the App**: Open the URL provided in the terminal (usually `http://localhost:8501`).

---

## **8. Fetching Real Data (YouTube)**

To use the "Fetch from YouTube" feature in the app:

1.  **Get a YouTube Data API Key**:
    - Go to [Google Cloud Console](https://console.cloud.google.com/).
    - Create a new project.
    - Enable **YouTube Data API v3**.
    - Create Credentials (API Key).
2.  **In the App**:
    - Select "Fetch from YouTube" in the sidebar.
    - Paste your API Key.
    - Enter a Video ID (e.g., `aq5nS9HkCGY` for Samsung S24 Ultra).
    - Click "Fetch Comments".

---

## **9. Future Improvements**

- **Advanced Models**: Replace TF-IDF/K-Means with BERTopic or LDA for better topic coherence.
- **Deep Learning Sentiment**: Use RoBERTa or DistilBERT for more nuanced sentiment detection (sarcasm, context).
- **Real-time API**: Connect directly to YouTube/Instagram APIs to fetch live comments.
- **Named Entity Recognition (NER)**: Automatically detect product names or competitor mentions.
