import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import plotly.express as px

# Set page config
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎭", layout="centered")

# Download NLTK data (only if not already downloaded)
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')

download_nltk_data()

# Custom CSS for styling
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Title styling */
    h1 {
        color: #2c3e50;
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        border-radius: 12px !important;
        border: 2px solid #e0e0e0 !important;
        padding: 15px !important;
        font-size: 16px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    .stTextArea textarea:focus {
        border-color: #3498db !important;
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.2);
    }
    
    /* Button styling */
    .stButton button {
        border-radius: 25px !important;
        background: linear-gradient(135deg, #6dd5ed, #2193b0) !important;
        color: white !important;
        font-weight: bold !important;
        font-size: 18px !important;
        padding: 10px 24px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(33, 147, 176, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(33, 147, 176, 0.4) !important;
    }
    
    /* Result box styling */
    .result-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-size: 28px;
        font-weight: 800;
        margin-top: 30px;
        margin-bottom: 20px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.08);
        animation: fadeIn 0.5s ease-in-out;
    }
    .positive {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        color: #1e5631;
    }
    .negative {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 99%, #fecfef 100%);
        color: #721c24;
    }
    .neutral {
        background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
        color: #2c3e50;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# App Title and Description
st.title("✨ Sentiment Analyzer")
st.markdown("<p style='text-align: center; font-size: 18px; color: #7f8c8d; margin-bottom: 30px;'>Discover the hidden emotions in your text</p>", unsafe_allow_html=True)

# User Input
user_text = st.text_area("", height=150, placeholder="Type or paste your text here to analyze its sentiment...")

# Analyze Button
if st.button("Analyze Sentiment", use_container_width=True):
    if user_text.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner('Analyzing sentiment...'):
            # Initialize the analyzer
            sia = SentimentIntensityAnalyzer()
            
            # Get sentiment scores
            sentiment_scores = sia.polarity_scores(user_text)
            compound_score = sentiment_scores['compound']
            
            # Determine sentiment category
            if compound_score >= 0.05:
                sentiment = "Positive"
                css_class = "positive"
                emoji = "😄"
            elif compound_score <= -0.05:
                sentiment = "Negative"
                css_class = "negative"
                emoji = "😞"
            else:
                sentiment = "Neutral"
                css_class = "neutral"
                emoji = "😐"
                
            # Display Result
            st.markdown(f"""
                <div class='result-box {css_class}'>
                    {emoji} {sentiment}
                </div>
            """, unsafe_allow_html=True)
            
            # Display detailed scores using a Plotly chart
            st.markdown("<h3 style='text-align: center; color: #2c3e50; margin-top: 30px;'>Confidence Scores</h3>", unsafe_allow_html=True)
            
            # Format data for chart
            scores_df = pd.DataFrame({
                'Sentiment': ['Negative', 'Neutral', 'Positive'],
                'Score': [sentiment_scores['neg'], sentiment_scores['neu'], sentiment_scores['pos']]
            })
            
            # Create a horizontal bar chart
            fig = px.bar(scores_df, x='Score', y='Sentiment', orientation='h',
                         color='Sentiment',
                         color_discrete_map={
                             'Negative': '#ff7675',
                             'Neutral': '#74b9ff',
                             'Positive': '#55efc4'
                         })
            
            # Update chart layout for cleaner look
            fig.update_layout(
                showlegend=False, 
                xaxis_range=[0, 1],
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Confidence Score (0 to 1)",
                yaxis_title="",
                height=250,
                margin=dict(l=0, r=0, t=10, b=0)
            )
            
            # Render chart
            st.plotly_chart(fig, use_container_width=True)
