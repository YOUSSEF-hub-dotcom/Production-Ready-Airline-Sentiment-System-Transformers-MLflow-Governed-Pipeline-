import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(page_title="Airline Sentiment AI Pro", page_icon="✈️", layout="wide")

API_BASE_URL = "http://127.0.0.1:8000"


# Function to create a Gauge Chart for AI confidence levels
def plot_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': "AI Confidence %", 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#007bff"},
            'steps': [
                {'range': [0, 40], 'color': "#ffcccc"},
                {'range': [40, 70], 'color': "#ffffcc"},
                {'range': [70, 100], 'color': "#ccffcc"}
            ],
        }
    ))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
    return fig


# Helper function to load and display global insights history seamlessly
def load_and_display_history():
    try:
        # Fetch historical prediction logs from the FastAPI backend
        history_res = requests.get(f"{API_BASE_URL}/predictions")
        if history_res.status_code == 200:
            history_data = history_res.json()
            if history_data:
                df = pd.DataFrame(history_data)

                c1, c2 = st.columns([1, 2])

                # Visualization: Overall Sentiment Distribution (Pie Chart)
                with c1:
                    st.write("**Overall Sentiment Distribution**")
                    sentiment_counts = df['sentiment'].value_counts()
                    fig_pie = go.Figure(
                        data=[go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values, hole=.3)])
                    fig_pie.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
                    st.plotly_chart(fig_pie, use_container_width=True)

                # Table view for the most recent predictions (Clean formatting)
                with c2:
                    st.write("**Recent Predictions (Latest Logs)**")
                    # Formatting created_at to be highly readable for end users
                    if 'created_at' in df.columns:
                        df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Showing head(10) instead of tail to always present the freshest insights first
                    st.dataframe(df[['text', 'sentiment', 'confidence', 'created_at']].head(10),
                                 use_container_width=True)
            else:
                st.info("ℹ️ No data available in history yet. Run some predictions first!")
        else:
            st.error("❌ Failed to communicate with history service endpoint.")
    except Exception as e:
        st.error(f"⚠️ Could not load history. Make sure the FastAPI server is running at {API_BASE_URL}")


# 2. Main UI Components
st.title("✈️ Airline Sentiment Analysis")
st.markdown("Analyze customer feedback in real-time with high-precision AI pipeline.")

# Sidebar Configuration for Dashboard Information
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/784/784306.png", width=100)
    st.title("Dashboard Info")
    st.info("Sentiment Analysis helps airlines improve customer service by identifying pain points.")
    st.divider()
    st.caption("Model Version: v3.0 (Production DistilBERT)")
    st.caption("Status: API Public Access Active")

# 3. Input Area and Real-time Analysis Section
st.subheader("📝 Analyze Tweet")
tweet_text = st.text_area(
    "✏️ Enter tweet text:",
    placeholder="Example: The flight was amazing and the crew was very helpful!",
    height=120
)

# Execution logic when the Analyze button is clicked
if st.button("🚀 Analyze Sentiment", use_container_width=True, type="primary"):
    if not tweet_text.strip():
        st.warning(" Please enter some text first.")
    else:
        with st.spinner("🧠 AI is processing..."):
            try:
                # Send a POST request to the FastAPI backend
                res = requests.post(f"{API_BASE_URL}/predict", json={"text": tweet_text})

                if res.status_code == 200:
                    data = res.json()
                    st.divider()

                    # Layout for displaying metrics and the gauge chart
                    col_metrics, col_chart = st.columns([1, 1])

                    with col_metrics:
                        sentiment = data["sentiment"]
                        # Display visual feedback based on sentiment type
                        if sentiment.lower() == "positive":
                            st.success(f"### Result: {sentiment.upper()} 😊")
                            st.balloons()
                        elif sentiment.lower() == "negative":
                            st.error(f"### Result: {sentiment.upper()} 😡")
                        else:
                            st.warning(f"### Result: {sentiment.upper()} 😐")

                        st.metric("Inference Speed", f"{data['inference_time']}s")
                        st.write("**AI Confidence:** Our fine-tuned model is highly confident in this classification.")

                    with col_chart:
                        st.plotly_chart(plot_gauge(data["confidence"]), use_container_width=True)

                elif res.status_code == 429:
                    st.error(" Rate limit exceeded! Please wait a minute.")
                else:
                    st.error(f" Error: {res.json().get('detail', 'Server Error')}")
            except Exception as e:
                st.error(f" Connection failed: {e}")

# 4. Global Insights & History Section
st.divider()
st.subheader(" Global Insights & History")

# Automatically display history when page loads for instant analytics insight
load_and_display_history()

# Manual refresh option to load newly streamed data
if st.button(" Refresh Analytics History", use_container_width=True):
    st.rerun()

# 5. Footer Information
st.divider()
st.caption("Developed for Airline Sentiment Enterprise Project | 2026")
