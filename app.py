import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import requests

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Restaurant Survival AI",
    layout="wide",
    page_icon="🍽️"
)

# Clean chart theme with transparent backgrounds to blend into cards
pio.templates.default = "simple_white"

# ----------------------------
# GLOBAL STYLE
# ----------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono&display=swap');

    /* Apply Space Grotesk to almost everything */
    html, body, [class*="st-"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    /* Apply Space Mono to code, metrics, and data for that tech look */
    code, pre, [data-testid="stMetricValue"], .stDataFrame {
        font-family: 'Space Mono', monospace !important;
    }

    /* Slight padding adjustment for a cleaner look */
    .block-container {
        padding-top: 1.5rem;
        max-width: 1200px;
    }

    /* Hero image styling */
    .hero-img img {
        border-radius: 18px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.12);
    }

    /* Centralize Tabs */
    div[data-testid="stTabs"] [role="tablist"] {
        justify-content: center;
    }

    /* Optional: Ensure the tabs don't stretch too wide */
    div[data-testid="stTabs"] button[role="tab"] {
        flex: 0 1 auto;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Hero Section Styling */
    .hero-container {
        text-align: center;
        padding: 2rem 0;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        margin-bottom: 2rem;
    }

    .main-title {
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 0px !important;
        color: #1E1E1E;
        letter-spacing: -1px;
    }

    .sub-title {
        font-size: 1.2rem !important;
        color: #666;
        margin-top: 0px !important;
    }

    /* ---------------------------------------------------- */
    /* INCREASE TAB FONT SIZE HERE                          */
    /* ---------------------------------------------------- */
    div[data-testid="stTabs"] button[role="tab"] p {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
    }

</style>
""", unsafe_allow_html=True)


# ----------------------------
# HELPERS
# ----------------------------
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.strip()

def get_sentiment_label(score):
    if score > 0.25: return "Positive"
    if score < -0.25: return "Negative"
    return "Neutral"

def score_to_heatmap_matrix(score):
    normalized = (score + 1) / 2
    return np.array([
        [normalized * 0.8, normalized * 0.9, normalized],
        [normalized * 0.7, normalized * 0.85, normalized * 0.95],
        [normalized * 0.65, normalized * 0.8, normalized * 0.9]
    ])

def generate_ai_explanation(sentiment, rating, review_count):
    explanations = []
    if sentiment < -0.3: explanations.append("Low overall sentiment highlights major customer dissatisfaction areas.")
    if rating < 3: explanations.append("Below-average star rating is an immediate red flag for foot traffic.")
    if review_count < 50: explanations.append("Low review volume indicates poor digital presence and visibility.")
    explanations.append("Combined metrics reflect an elevated operational risk in current market conditions.")
    return " ".join(explanations)

# Custom Gauge Chart Function
def rating_meter(value, title="Rating"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 14}},
        number={'font': {'size': 20, 'family': 'Space Mono'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#333"},
            'bar': {'color': "#BF5C3E"},
            'bgcolor': "white",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 33], 'color': "#F5F5F5"},
                {'range': [33, 66], 'color': "#E5E5E5"},
                {'range': [66, 100], 'color': "#D3D3D3"}
            ]
        }
    ))

    fig.update_layout(
        height=180,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'family': "Space Grotesk, sans-serif"}
    )

    return fig

# ----------------------------
# LOAD DATASET
# ----------------------------
@st.cache_data
def load_dataset():
    # Load dataset from the file uploaded
    df = pd.read_csv("dataset_rest_streamlit.csv")

    # Map 'business_stars' to 'avg_rating' so we don't break existing UI code
    df.rename(columns={"business_stars": "avg_rating"}, inplace=True)

    # Since we don't have true sentiment_score and closure_risk in the raw data,
    # let's generate appropriate simulated values to fulfill the UI logic.
    np.random.seed(42)

    # Generate a sentiment score closely correlated with the average rating (-1 to 1)
    df["sentiment_score"] = np.clip(np.random.normal((df["avg_rating"] - 3) / 2, 0.4), -1, 1)

    # Calculate simulated closure risk (higher risk for closed places 'is_open' == 0)
    df["closure_risk"] = np.where(
        df["is_open"] == 1,
        np.random.uniform(0.0, 0.49, len(df)),
        np.random.uniform(0.5, 1.0, len(df))
    )

    return df

df = load_dataset()

# ----------------------------
# HEADER
# ----------------------------

st.markdown('<div class="hero-img">', unsafe_allow_html=True)

try:
    st.image("header2.jpeg", use_container_width=True)
except:
    st.info("Header image placeholder (header2.jpeg not found)")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align:center;'>Behind The Stars</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Predict restaurant closure risk using early customer reviews and sentiment signals.</p>",
    unsafe_allow_html=True
)

st.divider()
tab1, tab2 = st.tabs(["Dataset Analysis", "Prediction Lab"])

# ----------------------------
# DATASET TAB
# ----------------------------
with tab1:
    with st.container(border=True):
        st.subheader("Dataset Overview")
        col1, col2, col3, col4, col5 = st.columns(5)

        # Adjusting the dynamic metrics calculation based on actual dataset loaded
        total_rests = len(df) / 1000
        total_revs = df.review_count.sum() / 1000000

        col1.metric("Total Restaurants", f"{total_rests:.1f}k")
        col2.metric("Total Reviews", "2.79M")
        col3.metric("Avg Rating", round(df.avg_rating.mean(), 2))
        col4.metric("Closure Risk Avg /holder/", round(df.closure_risk.mean(), 2))

        avg_reviews_per_rest = df.review_count.sum() / len(df)
        col5.metric("Review Density (Avg/Rest)", round(avg_reviews_per_rest, 1))

    col1, col2 = st.columns([2, 1])

    with col1:
        with st.container(border=True):
            st.markdown("<h4 style='text-align: center; margin-bottom: 0;'>Rating Distribution</h4>", unsafe_allow_html=True)

            bins = np.linspace(1, 5, 25)

            fig = go.Figure()

            # Closed
            fig.add_trace(go.Histogram(
                x=df[df['is_open'] == 0]["avg_rating"],
                xbins=dict(start=1, end=5, size=0.5),
                marker_color="gray",
                opacity=0.6,
                name="Closed"
            ))

            # Open
            fig.add_trace(go.Histogram(
                x=df[df['is_open'] == 1]["avg_rating"],
                xbins=dict(start=1, end=5, size=0.5),
                marker_color="#BF5C3E",
                opacity=0.6,
                name="Open"
            ))
            fig.update_layout(
                barmode="overlay",   # 👈 keep overlay
                bargap=0,
                bargroupgap=0,
                xaxis_title="Average Rating",
                yaxis_title="Frequency",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=20),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.15,
                    xanchor="center",
                    x=0.5
                )
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        with st.container(border=True):
            st.markdown("<h4 style='text-align: center; margin-bottom: 0;'>Open vs Closed</h4>", unsafe_allow_html=True)

            status = df.copy()
            status["status"] = np.where(status.is_open == 1, "Open", "Closed")

            fig = px.pie(
                status, names="status", hole=0.4,
                color="status", color_discrete_map={"Open": "#BF5C3E", "Closed": "#D3D3D3"}
            )

            fig.update_traces(
                hovertemplate="%{percent:.2f}%<extra></extra>"
            )

            fig.update_layout(
                barmode="overlay",
                bargap=0,
                bargroupgap=0,
                xaxis_title="Average Rating",
                yaxis_title="Frequency",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=20),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.15,
                    xanchor="center",
                    x=0.5
                )
            )

            st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        st.subheader("Restaurants Distribution Map")

        # Using built-in px.scatter_map for MapBox mapping
        fig_map = px.scatter_map(
            df,
            lat="latitude",
            lon="longitude",
            color="closure_risk",
            hover_name="name",
            zoom=3,
            size_max=15,
            color_continuous_scale=[
                "#E5B7A9",
                "#D48E75",
                "#BF5C3E"
            ]
        )

        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)

# ----------------------------
# PREDICTION TAB
# ----------------------------

with tab2:


    with st.container(border=True):
        st.subheader("Upload Dataset")
        csv_file = st.file_uploader(
            "Upload CSV File",
            type=["csv"],
            help="Upload a CSV file containing restaurant data"
        )

        if csv_file is not None:
            # 1. Read the uploaded file
            upload_df = pd.read_csv(csv_file)

            # 2. Define the exact columns you want to extract (keeps it flexible for later)
            target_cols = ['text_cleaned', 'business_id', 'date']

            # Check if those columns actually exist in the uploaded CSV
            if set(target_cols).issubset(upload_df.columns):
                # Extract the columns and convert to a list of dictionaries for the API
                extracted_data = upload_df[target_cols].to_dict(orient="records")

                # Placeholder URL for your API
                API_URL = "https://webhook.site/5821bd96-72fd-4e6d-833d-21dabe5b326f"

                # Send button
                if st.button("Send Data to API", type="primary"):
                    with st.spinner("Sending data..."):
                        try:
                            # Send the parameters as a JSON payload
                            response = requests.post(API_URL, json={"reviews": extracted_data})

                            if response.status_code == 200:
                                st.success("Data successfully sent to the API!")
                                # You can capture the response here later: response_data = response.json()
                            else:
                                st.error(f"API Error: {response.status_code} - {response.text}")
                        except Exception as e:
                            st.error(f"Failed to connect to the API. Error: {e}")
            else:
                st.warning(f"Missing required columns. Please ensure your CSV has: {target_cols}")

    review_text = "The food was cold and the service was slow"
    rating = 4
    review_count = 500
    sentiment = TextBlob(clean_text(review_text)).sentiment.polarity
    label = get_sentiment_label(sentiment)
    heat = score_to_heatmap_matrix(sentiment)
    risk = max(0, min(1, (sentiment < -0.3)*0.3 + (rating < 3)*0.3 + (review_count < 50)*0.2))

    with st.container(border=True):
        st.subheader("Real-time Analysis")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sentiment Score", round(sentiment, 2))
        c2.metric("Sentiment Label", label)
        c3.metric("Predicted Closure Risk", f"{risk*100:.1f}%")

        with c4:
            st.progress(risk, text="Risk Level")

    # Topic Meters Section using the new Plotly Gauge
    topic_labels = {
        0: "food_quality", 1: "service", 2: "waiting_time", 3: "price_value",
        4: "order_accuracy", 5: "cleanliness", 6: "atmosphere",
        7: "location_access", 8: "management", 9: "portion_size"
    }

    with st.container(border=True):
        st.subheader("Review Topic Analysis")
        st.caption("Extracted sentiment drivers from customer reviews.")

        row1 = st.columns(5)
        row2 = st.columns(5)

        for i, (key, topic) in enumerate(topic_labels.items()):
            col = row1[i] if i < 5 else row2[i - 5]
            with col:
                mock_score = np.random.randint(40, 95)
                clean_name = topic.replace("_", " ").title()
                fig_gauge = rating_meter(mock_score, title=clean_name)
                st.plotly_chart(fig_gauge, use_container_width=True, key=f"gauge_{i}")

    with st.container(border=True):
        st.subheader("🔍 Similar Restaurants")
        st.caption("Based on review analysis and operational metrics.")
        st.markdown("- **The Golden Spoon** (Similarity: 92%) - *Matches low sentiment on 'waiting time'*")
        st.markdown("- **Bistro 101** (Similarity: 85%) - *Matches average rating and review volume*")
        st.markdown("- **Downtown Diner** (Similarity: 78%) - *Similar risk profile and location*")

    with st.container(border=True):
        st.subheader("💡 AI Insight")
        explanation = generate_ai_explanation(sentiment, rating, review_count)
        st.info(explanation, icon="🧠")
