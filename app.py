import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go  # <-- Added for the gauge charts
import plotly.io as pio
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

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
        font-size: 1.5rem !important;  /* Change this value to adjust size (e.g., 24px) */
        font-weight: 600 !important;   /* Optional: makes the text slightly bolder */
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
    if sentiment < -0.3: explanations.append("Customer sentiment is strongly negative.")
    if rating < 3: explanations.append("Low average rating indicates dissatisfaction.")
    if review_count < 50: explanations.append("Low review volume suggests weak traction.")
    explanations.append("Restaurants showing declining sentiment early often experience business risk.")
    return " ".join(explanations)

# New Custom Gauge Chart Function
def rating_meter(value, title="Rating"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 14}},
        number={'font': {'size': 20, 'family': 'Space Mono'}}, # Tech font for numbers
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#333"},
            'bar': {'color': "#BF5C3E"}, # Matches the brand highlight color
            'bgcolor': "white",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 33], 'color': "#F5F5F5"},
                {'range': [33, 66], 'color': "#E5E5E5"},
                {'range': [66, 100], 'color': "#D3D3D3"} # Matches neutral gray
            ]
        }
    ))

    fig.update_layout(
        height=180, # Compact height to fit nicely in 5 columns
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", # Transparent background
        plot_bgcolor="rgba(0,0,0,0)",
        font={'family': "Space Grotesk, sans-serif"} # Matches app font
    )

    return fig

# ----------------------------
# MOCK DATASET
# ----------------------------
@st.cache_data
def generate_mock_dataset(n=1200):

    # Major US restaurant hubs (lat, lon)
    cities = [
        ("New York", 40.7128, -74.0060),
        ("Los Angeles", 34.0522, -118.2437),
        ("Chicago", 41.8781, -87.6298),
        ("Houston", 29.7604, -95.3698),
        ("Phoenix", 33.4484, -112.0740),
        ("Philadelphia", 39.9526, -75.1652),
        ("San Antonio", 29.4241, -98.4936),
        ("San Diego", 32.7157, -117.1611),
        ("Dallas", 32.7767, -96.7970),
        ("San Francisco", 37.7749, -122.4194),
        ("Seattle", 47.6062, -122.3321),
        ("Miami", 25.7617, -80.1918),
        ("Atlanta", 33.7490, -84.3880),
        ("Boston", 42.3601, -71.0589),
        ("Denver", 39.7392, -104.9903),
        ("Las Vegas", 36.1699, -115.1398),
        ("Washington DC", 38.9072, -77.0369),
        ("Austin", 30.2672, -97.7431),
        ("Nashville", 36.1627, -86.7816),
        ("Orlando", 28.5383, -81.3792)
    ]

    rows = []

    for i in range(n):

        # Pick a city
        city, lat_c, lon_c = cities[np.random.randint(len(cities))]

        # Generate location near the city
        lat = lat_c + np.random.normal(0, 0.25)
        lon = lon_c + np.random.normal(0, 0.25)

        rating = np.clip(np.random.normal(3.5, 0.9), 1, 5)
        sentiment = np.clip(np.random.normal((rating-3)/2, 0.4), -1, 1)
        review_count = np.random.randint(5, 400)

        risk = max(
            0,
            min(1, (sentiment < -0.3)*0.3 + (rating < 3)*0.3 + (review_count < 50)*0.2)
        )

        rows.append({
            "name": f"Restaurant {i}",
            "city": city,
            "latitude": lat,
            "longitude": lon,
            "avg_rating": rating,
            "review_count": review_count,
            "sentiment_score": sentiment,
            "closure_risk": risk,
            "is_open": int(risk < 0.55)
        })

    return pd.DataFrame(rows)

df = generate_mock_dataset()

# ----------------------------
# HEADER
# ----------------------------

st.markdown('<div class="hero-img">', unsafe_allow_html=True)

try:
    st.image("header2.jpeg", use_container_width=True)
except:
    st.info("Header image placeholder (header.png not found)")

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
        col1.metric("Total Restaurants", len(df))
        col2.metric("Total Reviews", df.review_count.sum())
        col3.metric("Avg Rating", round(df.avg_rating.mean(), 2))
        col4.metric("Closure Risk Avg", round(df.closure_risk.mean(), 2))

        # Replaced KPI X with Review Density here
        avg_reviews_per_rest = df.review_count.sum() / len(df)
        col5.metric("Review Density (Avg/Rest)", round(avg_reviews_per_rest, 1))

    col1, col2 = st.columns([2, 1])

    with col1:
        with st.container(border=True):
            st.markdown("<h4 style='text-align: center; margin-bottom: 0;'>Rating Distribution</h4>", unsafe_allow_html=True)

            # --- Prepare aligned bins ---
            bins = np.linspace(1, 5, 25)

            fig = go.Figure()

            # Rating distribution (gray)
            fig.add_trace(go.Histogram(
                x=df["avg_rating"],
                xbins=dict(start=1, end=5, size=(5-1)/25),
                marker_color="gray",
                opacity=0.65,
                name="Closed"
            ))

            # Sentiment distribution (scaled to rating range)
            scaled_sentiment = ((df["sentiment_score"] + 1) / 2) * 4 + 1

            fig.add_trace(go.Histogram(
                x=scaled_sentiment,
                xbins=dict(start=1, end=5, size=(5-1)/25),
                marker_color="#BF5C3E",
                opacity=0.65,
                name="Open"
            ))

            fig.update_layout(
                barmode="overlay",
                xaxis_title="Score",
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

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=20),

                legend=dict(
                    orientation="h",   # horizontal legend
                    yanchor="top",
                    y=-0.1,            # position below chart
                    xanchor="center",
                    x=0.5
                )
            )

            st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        st.subheader("Restaurants Distribution Map")
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
                "#BF5C3E"   # brand color
            ]
        )

        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)

# ----------------------------
# PREDICTION TAB
# ----------------------------
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("Restaurant Inputs")
            city = st.text_input("City", "Las Vegas")
            rating = st.slider("Average Rating", 1.0, 5.0, 3.0)
            review_count = st.slider("Review Count", 1, 500, 100)

    with col2:
        with st.container(border=True):
            st.subheader("Customer Feedback")
            review_text = st.text_area(
                "Customer Review",
                "The food was cold and the service was slow.",
                height=130
            )

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
                # Generate a mock score between 0 and 100 for the gauge
                mock_score = np.random.randint(40, 95)
                clean_name = topic.replace("_", " ").title()

                # Render the custom Plotly gauge chart
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
