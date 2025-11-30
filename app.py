import streamlit as st
import pandas as pd
import numpy as np
import os

# Import helper functions from nlp_utils for cleaning, sentiment, and topic modeling
from nlp_utils import clean_text, get_sentiment, perform_topic_modeling

# ---------------------------
# Streamlit App Configuration
# ---------------------------

st.set_page_config(
    page_title="Brand NLP Insights",
    layout="wide",
    page_icon="🤖"
)

def load_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Orbitron:wght@500&display=swap');

        /* Main Background - Clean Dark */
        .stApp {
            background-color: #222831;
        }

        /* Typography */
        h1, h2, h3 {
            font-family: 'Orbitron', sans-serif !important;
            color: #00ADB5 !important;
            font-weight: 500;
        }
        
        .stMarkdown, p, .stText, .stDataFrame {
            font-family: 'Inter', sans-serif !important;
            color: #EEEEEE;
        }

        /* Sidebar - Professional Dark Grey */
        section[data-testid="stSidebar"] {
            background-color: #393E46;
            border-right: 1px solid #00ADB5;
        }

        /* Buttons - Minimalist */
        .stButton > button {
            background-color: transparent !important;
            border: 1px solid #00ADB5 !important;
            color: #00ADB5 !important;
            border-radius: 5px !important;
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            background-color: #00ADB5 !important;
            color: #ffffff !important;
        }

        /* Inputs */
        .stTextInput > div > div > input {
            color: #EEEEEE !important;
            border-bottom: 2px solid #00ADB5 !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 20px;
        }

        .stTabs [data-baseweb="tab"] {
            height: 40px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 0px;
            color: #aaa;
            font-family: 'Inter', sans-serif;
            border: none;
            border-bottom: 2px solid transparent;
        }

        .stTabs [aria-selected="true"] {
            background-color: transparent !important;
            color: #00ADB5 !important;
            border-bottom: 2px solid #00ADB5 !important;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            font-family: 'Inter', sans-serif !important;
            font-weight: 600;
        }
        
        </style>
    """, unsafe_allow_html=True)

load_css()

st.title("🧠 Brand Intel // Sentiment Analytics")

st.markdown(
    """
    <div style='background-color: #393E46; padding: 15px; border-radius: 5px; border-left: 4px solid #00ADB5; margin-bottom: 20px;'>
    <p style='margin:0; font-size: 1rem;'>
    <strong>System Active:</strong> Analyzing social media sentiment patterns and topic clusters.
    </p>
    </div>
    """, unsafe_allow_html=True
)

# ---------------------------
# Sidebar: Data Input & Settings
# ---------------------------
st.sidebar.header("Upload & Settings")

# User chooses where the data comes from
data_source = st.sidebar.radio(
    "Data Source",
    ("Upload CSV", "Use Sample Data", "Fetch from YouTube")
)

df = None

# Option 1: Upload a local CSV file
if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload comments CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

# Option 2: Use the provided synthetic sample data
elif data_source == "Use Sample Data":
    sample_path = os.path.join(os.path.dirname(__file__), "data", "sample_comments.csv")
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
        st.sidebar.success("Loaded sample data!")
    else:
        st.sidebar.error("Sample data file not found.")

# Option 3: Fetch real data from YouTube using the API
elif data_source == "Fetch from YouTube":
    st.sidebar.markdown("### YouTube Settings")
    
    # Try to load API key from .streamlit/secrets.toml first
    # This prevents the user from having to enter it every time
    if "YOUTUBE_API_KEY" in st.secrets and st.secrets["YOUTUBE_API_KEY"] != "REPLACE_WITH_YOUR_ACTUAL_API_KEY":
        api_key = st.secrets["YOUTUBE_API_KEY"]
        st.sidebar.success("API Key loaded from secrets ✅")
    else:
        # Fallback: Ask user to enter key manually
        api_key = st.sidebar.text_input("Enter YouTube API Key", type="password", help="Add your key to .streamlit/secrets.toml to avoid entering it here.")
    
    # Input for YouTube Video URL
    youtube_url = st.sidebar.text_input("YouTube Video URL", value="https://www.youtube.com/watch?v=aq5nS9HkCGY")
    
    if st.sidebar.button("Fetch Comments"):
        if not api_key:
            st.sidebar.error("Please enter an API Key.")
        elif not youtube_url:
            st.sidebar.error("Please enter a YouTube URL.")
        else:
            # Logic to extract Video ID from various YouTube URL formats
            video_id = None
            if "v=" in youtube_url:
                video_id = youtube_url.split("v=")[1].split("&")[0]
            elif "youtu.be/" in youtube_url:
                video_id = youtube_url.split("youtu.be/")[1].split("?")[0]
            
            if not video_id:
                st.sidebar.error("Could not extract Video ID. Please check the URL.")
            else:
                try:
                    from googleapiclient.discovery import build
                    
                    with st.spinner(f"Fetching comments for Video ID: {video_id}..."):
                        # Initialize YouTube API client
                        youtube = build('youtube', 'v3', developerKey=api_key)
                        
                        comments_data = []
                        try:
                            # Request comments from the video
                            request = youtube.commentThreads().list(
                                part="snippet",
                                videoId=video_id,
                                maxResults=100, # Fetch up to 100 per page
                                textFormat="plainText"
                            )
                            
                            # Pagination loop to fetch up to 500 comments
                            while request and len(comments_data) < 500: 
                                response = request.execute()
                                for item in response.get("items", []):
                                    comment = item["snippet"]["topLevelComment"]["snippet"]
                                    comments_data.append({
                                        "comment_text": comment.get("textDisplay"),
                                        "likes": comment.get("likeCount"),
                                        "published_at": comment.get("publishedAt"),
                                        "platform": "YouTube"
                                    })
                                
                                # Check if there are more pages
                                if "nextPageToken" in response:
                                    request = youtube.commentThreads().list(
                                        part="snippet",
                                        videoId=video_id,
                                        maxResults=100,
                                        textFormat="plainText",
                                        pageToken=response["nextPageToken"]
                                    )
                                else:
                                    break
                                    
                            if comments_data:
                                df = pd.DataFrame(comments_data)
                                st.sidebar.success(f"Fetched {len(df)} comments!")
                            else:
                                st.sidebar.warning("No comments found or comments are disabled.")
                        except Exception as api_error:
                             st.sidebar.error(f"YouTube API Error: {api_error}")

                except Exception as e:
                    st.sidebar.error(f"Error: {e}")

# Additional Filters
platform = st.sidebar.selectbox(
    "Filter by Platform",
    ["All", "Instagram", "YouTube", "X (Twitter)", "Other"]
)

# Topic Modeling Parameter
n_topics = st.sidebar.slider(
    "Number of topics (clusters)",
    min_value=2,
    max_value=10,
    value=5
)

st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ About this Project"):
    st.markdown("""
    **What does this tool do?**
    
    1.  **Fetches Data**: Pulls real comments from YouTube videos or uses your own CSV data.
    2.  **Analyzes Text**: Uses Natural Language Processing (NLP) to understand human language.
    3.  **Finds Insights**:
        *   **Sentiment**: Are people happy or angry? (Positive/Negative)
        *   **Topics**: What are they talking about? (e.g., "Price", "Camera", "Shipping")
    4.  **Recommends Strategy**: Tells brands what to fix or highlight based on the data.
    
    **Tech Stack**: Python, Streamlit, TextBlob, Scikit-learn.
    """)

# ---------------------------
# Main App Logic
# ---------------------------

if df is not None:
    # Validation: Ensure the dataset has the required column
    if "comment_text" not in df.columns:
        st.error("CSV must contain a 'comment_text' column.")
    else:
        # Apply Platform Filter
        if platform != "All" and "platform" in df.columns:
            df = df[df["platform"].str.contains(platform.split()[0], case=False, na=False)]

        st.subheader("Raw Data Preview")
        st.dataframe(df.head())

        # ---------------------------
        # NLP Processing Pipeline
        # ---------------------------
        with st.spinner("Processing Comments..."):
            # 1. Clean Text
            df["clean_text"] = df["comment_text"].apply(clean_text)

            # 2. Sentiment Analysis
            # Apply get_sentiment function to each comment
            sentiments = df["clean_text"].apply(get_sentiment)
            # Unpack results into two new columns
            df["sentiment_score"] = [s[0] for s in sentiments]
            df["sentiment_label"] = [s[1] for s in sentiments]

            # 3. Topic Modeling
            texts_for_topics = df["clean_text"].tolist()
            # Filter out empty strings to avoid errors
            texts_for_topics = [t for t in texts_for_topics if t.strip()]
            
            # Only run topic modeling if we have enough data
            if len(texts_for_topics) > n_topics:
                # Run the helper function
                clusters, topic_keywords, kmeans_model = perform_topic_modeling(
                    texts_for_topics, n_topics=n_topics
                )
                
                # Map clusters back to the original dataframe
                # We create a temporary dataframe of valid texts to align the clusters
                df_topics = df[df["clean_text"].str.strip() != ""].copy()
                # Re-run on the subset to ensure alignment (fast enough for this scale)
                clusters, topic_keywords, _ = perform_topic_modeling(df_topics["clean_text"].tolist(), n_topics=n_topics)
                
                if clusters is not None:
                    df_topics["topic_cluster"] = clusters
                    # Merge back to main df
                    df["topic_cluster"] = df_topics["topic_cluster"]
                    # Fill NaN for rows that were skipped (empty comments) with -1
                    df["topic_cluster"] = df["topic_cluster"].fillna(-1).astype(int)
            else:
                clusters = None
                topic_keywords = []
                st.warning("Not enough data for topic modeling.")

        # ---------------------------
        # Visualization Tabs
        # ---------------------------
        tab_overview, tab_sentiment, tab_topics, tab_keywords, tab_reco = st.tabs(
            ["Overview", "Sentiment Analysis", "Topic Explorer", "Keyword Insights", "Recommendations"]
        )

        # --- Tab 1: Overview ---
        with tab_overview:
            st.markdown("### Dataset Overview")
            st.write(f"Total comments: **{len(df)}**")
            if "timestamp" in df.columns:
                try:
                    st.write(f"Date range: **{df['timestamp'].min()}** to **{df['timestamp'].max()}**")
                except:
                    pass

            # Show simple bar chart of sentiment counts
            sentiment_counts = df["sentiment_label"].value_counts()
            st.bar_chart(sentiment_counts)

        # --- Tab 2: Sentiment Analysis ---
        with tab_sentiment:
            st.markdown("### Sentiment Breakdown")
            st.write("Distribution of comment sentiments.")

            # Ensure all categories are present for consistent plotting
            sentiment_counts = df["sentiment_label"].value_counts().reindex(
                ["Positive", "Neutral", "Negative"]
            ).fillna(0)

            st.bar_chart(sentiment_counts)

            # Filter functionality
            selected_sentiment = st.selectbox(
                "Filter comments by sentiment",
                ["All", "Positive", "Neutral", "Negative"]
            )

            if selected_sentiment != "All":
                filtered = df[df["sentiment_label"] == selected_sentiment]
            else:
                filtered = df

            st.write(f"Showing {len(filtered)} comments:")
            st.dataframe(filtered[["comment_text", "sentiment_label", "sentiment_score"]])

        # --- Tab 3: Topic Explorer ---
        with tab_topics:
            st.markdown("### Topic Clusters")

            if clusters is None:
                st.warning("Not enough data for topic modeling.")
            else:
                # Display each topic with its keywords and sample comments
                for i, keywords in enumerate(topic_keywords):
                    st.markdown(f"**Topic {i}:** {keywords}")
                    # Get comments belonging to this topic
                    topic_comments = df[df["topic_cluster"] == i]["comment_text"]
                    sample_comments = topic_comments.head(5)
                    with st.expander(f"View sample comments for Topic {i} ({len(topic_comments)} comments)"):
                        for c in sample_comments:
                            st.write(f"- {c}")

        # --- Tab 4: Keyword Insights ---
        with tab_keywords:
            st.markdown("### Keyword Insights")
            # Flatten all cleaned text into a single list of words
            all_words = " ".join(df["clean_text"].astype(str).tolist()).split()
            
            # Simple stopword removal for visualization
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            filtered_words = [w for w in all_words if w not in ENGLISH_STOP_WORDS and len(w) > 2]
            
            # Count frequency
            freq = pd.Series(filtered_words).value_counts().head(20)
            st.write("Top words in comments (excluding common stopwords):")
            st.bar_chart(freq) 
            st.dataframe(freq.to_frame("count"))

        # --- Tab 5: Recommendations ---
        with tab_reco:
            st.markdown("### Suggested Communication Actions")

            # Filter comments by sentiment
            negative_comments = df[df["sentiment_label"] == "Negative"]
            positive_comments = df[df["sentiment_label"] == "Positive"]

            st.write("#### 1. Pain Points to Address")
            # Identify which topics are most prevalent in negative comments
            if clusters is not None and len(negative_comments) > 0:
                if "topic_cluster" in negative_comments.columns:
                    neg_cluster_counts = negative_comments["topic_cluster"].value_counts()
                    # Remove the -1 cluster (noise/empty)
                    neg_cluster_counts = neg_cluster_counts[neg_cluster_counts.index != -1]
                    
                    if not neg_cluster_counts.empty:
                        for i in neg_cluster_counts.index[:3]:  # Top 3 negative topics
                            if i < len(topic_keywords):
                                st.write(f"- **Topic {i}** (keywords: *{topic_keywords[i]}*) has **{neg_cluster_counts[i]}** negative comments.")
                                st.write("  → Consider addressing these issues in FAQs or dedicated posts.\n")
                    else:
                        st.write("No specific topic clusters found in negative comments.")
            else:
                st.write("No negative comments found or topic modeling failed.")

            st.write("#### 2. Messaging That Works Well")
            # Identify which topics are most prevalent in positive comments
            if clusters is not None and len(positive_comments) > 0:
                if "topic_cluster" in positive_comments.columns:
                    pos_cluster_counts = positive_comments["topic_cluster"].value_counts()
                    pos_cluster_counts = pos_cluster_counts[pos_cluster_counts.index != -1]
                    
                    if not pos_cluster_counts.empty:
                        for i in pos_cluster_counts.index[:3]: # Top 3 positive topics
                            if i < len(topic_keywords):
                                st.write(f"- **Topic {i}** (keywords: *{topic_keywords[i]}*) has **{pos_cluster_counts[i]}** positive comments.")
                                st.write("  → Reinforce these themes and wording in future campaigns.\n")
                    else:
                         st.write("No specific topic clusters found in positive comments.")
            else:
                st.write("No positive comments found or topic modeling failed.")

            st.write("#### 3. General Recommendations")
            st.markdown(
                """
                - Monitor sentiment after each major post or campaign.
                - Create **response templates** for recurring complaints.
                - Highlight features or benefits that appear in **positive comments**.
                - Track changes in sentiment over time as communication strategy is updated.
                """
            )

else:
    st.info("Please upload a CSV file with a 'comment_text' column or use the sample data to begin.")
