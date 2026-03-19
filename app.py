import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from textblob import TextBlob
import re
import requests

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Behind the Stars",
    layout="wide",
    page_icon="🍽️"
)

# Clean chart theme with transparent backgrounds to blend into cards
pio.templates.default = "simple_white"

# Initialize session state for the API response and risk value
if 'api_risk' not in st.session_state:
    st.session_state.api_risk = None
if 'api_response' not in st.session_state:
    st.session_state.api_response = None

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

def text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.strip()


# Custom Gauge Chart Function
def rating_meter(value, title="Rating"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 14}},
        number={'font': {'size': 20, 'family': 'Space Mono'}, 'valueformat': '.1f'},
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
        height=200,
        margin=dict(l=10, r=10, t=50, b=10),
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
    try:
        df = pd.read_csv("dataset_rest_streamlit.csv")
    except FileNotFoundError:
        # Fallback empty dataframe if not found locally for testing
        df = pd.DataFrame(columns=["business_stars", "is_open", "review_count", "latitude", "longitude", "name"])
        return df

    # Map 'business_stars' to 'avg_rating' so we don't break existing UI code
    if "business_stars" in df.columns:
        df.rename(columns={"business_stars": "avg_rating"}, inplace=True)

    # Since we don't have true sentiment_score and closure_risk in the raw data,
    # let's generate appropriate simulated values to fulfill the UI logic.
    np.random.seed(42)

    if "avg_rating" in df.columns and "is_open" in df.columns:
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
    st.image("header10.png", use_container_width=True)
except:
    st.info("Header image placeholder (header2.jpeg not found)")

st.markdown('</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Prediction Lab", "Dataset Analysis"])

# ----------------------------
# DATASET TAB
# ----------------------------
with tab2:
    with st.container(border=True):
        st.subheader("Dataset Overview")
        col1, col2, col3, col5 = st.columns(4)

        if not df.empty and "review_count" in df.columns and "avg_rating" in df.columns and "closure_risk" in df.columns:
            # Adjusting the dynamic metrics calculation based on actual dataset loaded
            total_rests = len(df) / 1000
            total_revs = df.review_count.sum() / 1000000

            col1.metric("Total Restaurants", f"{total_rests:.1f}k")
            col2.metric("Total Reviews", f"{total_revs:.2f}M")
            col3.metric("Avg Rating", round(df.avg_rating.mean(), 2))

            avg_reviews_per_rest = df.review_count.sum() / len(df)
            col5.metric("Review Density (Avg/Rest)", round(avg_reviews_per_rest, 1))

    col1, col2 = st.columns([2, 1])

    with col1:
        with st.container(border=True):
            st.markdown("<h4 style='text-align: center; margin-bottom: 0;'>Rating Distribution</h4>", unsafe_allow_html=True)

            if not df.empty and 'is_open' in df.columns and 'avg_rating' in df.columns:
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

            if not df.empty and 'is_open' in df.columns:
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

        if not df.empty and 'latitude' in df.columns and 'longitude' in df.columns and 'closure_risk' in df.columns:
            # Using built-in px.scatter_map for MapBox mapping
            fig_map = px.scatter_map(
                df,
                lat="latitude",
                lon="longitude",
                color="closure_risk",
                hover_name="name" if "name" in df.columns else None,
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

with tab1:

    # Initialize a variable to track the business_id from the uploaded CSV
    current_biz_id = None

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
            target_cols = ['text', 'business_id', 'date', 'is_open']

            # Check if those columns actually exist in the uploaded CSV

            if set(target_cols).issubset(upload_df.columns):

                # 1. Clean the text so the API doesn't choke on weird list-like strings
                clean_df = upload_df[target_cols].copy()

                clean_df['Unnamed: 0'] = clean_df.index

                # 2. Convert the clean dataframe BACK to CSV bytes
                csv_bytes = clean_df.to_csv(index=False).encode('utf-8')

                API_URL = "https://behind-the-stars-866652447486.europe-west1.run.app/api_csv/"

                # Send button
                if st.button("Send Data to API", type="primary"):
                    with st.spinner("Sending file..."):
                        try:
                            # 3. CRITICAL FIX: Send as a FILE using multipart/form-data
                            # This attaches the CSV bytes to a form field named "data"
                            files = {
                                "data": ("dataset.csv", csv_bytes, "text/csv")
                            }

                            # Notice we use `files=` instead of `json=` and removed the headers
                            response = requests.post(API_URL, files=files)

                            if response.status_code == 200:
                                st.success("Data successfully sent to the API!")
                                response_data = response.json()

                                # Save the FULL response to session state
                                st.session_state.api_response = response_data

                                # Capture the API risk
                                # NEW LOGIC:
                                # 1. Safely get the 'proba' dictionary from the response
                                proba_dict = response_data.get("proba", {})

                                # 2. Check which key exists and apply the math
                                if "open likelyhood" in proba_dict:
                                    # If it's the probability of being OPEN, risk is 1 minus that value
                                    risk_val = 1 - float(proba_dict["open likelyhood"])
                                elif "closure likelyhood" in proba_dict:
                                    # If it's already the probability of CLOSURE, use it directly
                                    risk_val = float(proba_dict["closure likelyhood"])
                                else:
                                    # Fallback if neither key is found
                                    risk_val = 0.0

                                # 3. Save the result to session state
                                st.session_state.api_risk = risk_val

                            else:
                                st.error(f"API Error: {response.status_code}")
                                with st.expander("See Error Details"):
                                    st.write("**Full API Response:**")
                                    try:
                                        st.json(response.json())
                                    except:
                                        st.write(response.text)
                        except Exception as e:
                            st.error(f"Failed to connect to the API. Error: {e}")
            else:
                st.warning(f"Missing required columns. Please ensure your CSV has: {target_cols}")



    review_text = "The food was cold and the service was slow"
    rating = 4
    review_count = 500
    sentiment = TextBlob(text(review_text)).sentiment.polarity

    # Use API value if available, otherwise fallback to default calculation
    if st.session_state.api_risk is not None:
        risk = st.session_state.api_risk
    else:
        risk = max(0, min(1, (sentiment < -0.3)*0.3 + (rating < 3)*0.3 + (review_count < 50)*0.2))

    with st.container(border=True):
        st.subheader("Real-time Analysis")
        c3, c4 = st.columns(2)
        c3.metric("Predicted Closure Risk", f"{risk*100:.1f}%")

        with c4:
            # Ensure risk is strictly clamped between 0.0 and 1.0 for the progress bar
            progress_val = max(0.0, min(1.0, float(risk)))
            st.progress(progress_val, text="Risk Level")

    # Topic Meters Section using the new Plotly Gauge
    target_sentiment_keys = [
        "sentiment_food_quality", "sentiment_service", "sentiment_waiting_time",
        "sentiment_price_value", "sentiment_order_accuracy", "sentiment_cleanliness",
        "sentiment_atmosphere", "sentiment_location_access", "sentiment_management",
        "sentiment_portion_size"
    ]

    with st.container(border=True):
        st.subheader("Review Topic Analysis")
        st.caption("Sentiment drivers scaled from 0 (Negative) to 100 (Positive).")

        if st.session_state.api_response and "topic_dico" in st.session_state.api_response:
            topic_data = st.session_state.api_response["topic_dico"][0]

            # Filter for keys that exist in the JSON
            found_topics = [k for k in target_sentiment_keys if k in topic_data]

            # REMOVED THE [:2] SLICE HERE to display all found topics
            if found_topics:
                # Create exactly as many columns as there are topics
                cols = st.columns(len(found_topics))

                for i, topic_key in enumerate(found_topics):
                    with cols[i]:
                        # RESCALING: Convert -1/1 to 0/100
                        raw_score = float(topic_data[topic_key])
                        scaled_score = (raw_score + 1) * 50

                        # Clean up the name for the chart title
                        clean_name = topic_key.replace("sentiment_", "").replace("_", " ").title()

                        # Generate and render the gauge chart
                        fig_gauge = rating_meter(scaled_score, title=clean_name)
                        st.plotly_chart(fig_gauge, use_container_width=True, key=f"gauge_{topic_key}")
            else:
                st.info("No specific sentiment topics found.")
        else:
            st.info("Upload a dataset to see the sentiment analysis.")

    with st.container(border=True):
        st.subheader("🔍 Similar Restaurants")
        st.caption("Based on review analysis and operational metrics.")

        # 1. Access the 'recommandation' key from your FASTAPI JSON
        if st.session_state.api_response and "recommandation" in st.session_state.api_response:
            recs = st.session_state.api_response["recommandation"]

            if recs:
                for item in recs:
                    # 2. Extract the name and similarity score
                    rest_name = item.get("name", "Unknown Restaurant")
                    # Similarity is usually 0-1, so we multiply by 100 for display
                    sim_score = item.get("similarity", 0) * 100

                    # 3. Create a clean display row
                    st.markdown(f"""
                    <div style="padding: 10px; border-bottom: 1px solid #f0f2f6;">
                        <strong>{rest_name}</strong> <br>
                        <span style="color: #BF5C3E; font-family: 'Space Mono'; font-size: 0.9rem;">
                            Match Score: {sim_score:.1f}%
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No similar restaurants found.")
        else:
            st.info("Upload a dataset and send it to the API to see similar restaurants.")

# ----------------------------
#    # DEBUG: RAW API RESPONSE
#    # ----------------------------
#    st.markdown("---") # Adds a nice visual divider
#    with st.expander("🛠️ View Raw API Response JSON"):
#        if st.session_state.api_response:
#            # st.json automatically formats dictionaries beautifully
#            st.json(st.session_state.api_response)
#        else:
#            st.info("No API data yet. Upload a CSV and click 'Send Data to API' to see the JSON payload.")
