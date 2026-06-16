import streamlit as st
import pickle
import re
import string
import pandas as pd
import plotly.express as px
def clean_tweet(text):
    text=str(text)
    text=re.sub(r'@[A-Za-z0-9_]+', '', text)
    text=re.sub(r'https?://\S+', '', text)
    text=re.sub(r'#', '', text)
    text=''.join([char for char in text if char not in string.punctuation])
    return text.lower().strip()

st.set_page_config(page_title="Social Media Sentiment Analyzer",page_icon="📝",layout="wide")

@st.cache_resource
def load_all_assets():
    with open('vectorizer.pkl','rb') as f:
        vectorizer=pickle.load(f)
    with open('naive_bayes_model.pkl','rb') as f:
        nb_model=pickle.load(f)
    with open('logistic_model.pkl', 'rb') as f:
        lr_model=pickle.load(f)
    return vectorizer,nb_model,lr_model

vectorizer,nb_model,lr_model = load_all_assets()

st.sidebar.title("📊 NLP ANALYZER")
st.sidebar.markdown("---")
st.sidebar.subheader("Configuration")

model_choice = st.sidebar.selectbox("Choose Classification Engine:", ["Multinomial Naive Bayes", "Logistic Regression"])

if model_choice=="Multinomial Naive Bayes":
    active_model=nb_model
else:
    active_model=lr_model

st.sidebar.markdown("---")
st.sidebar.subheader("About this Project")
st.sidebar.write(
    "This dashboard is powered by the Sentiment Dataset. "
    "It implements token matching algorithms across various pipeline structures."
)
st.sidebar.markdown("---")
st.sidebar.caption("Made with Streamlit")

st.title("💬Social Media Sentiment Analysis Dashboard")
st.info(f"**Currently Evaluating Using Engine:** {model_choice}")

user_tweet = st.text_area("Enter Text to Analyze:", placeholder="Type here...")

if st.button("Run Sentiment Analysis"):
    if user_tweet.strip() != "":
        cleaned = clean_tweet(user_tweet)
        vectorized_text = vectorizer.transform([cleaned])
        
        prediction = active_model.predict(vectorized_text)
        probabilities = active_model.predict_proba(vectorized_text)
        confidence_percentage = max(probabilities[0]) * 100

        st.markdown("---")
        if prediction == 1:
            st.success(f"### 😄 Sentiment Outcome: POSITIVE  |  Engine Confidence Score: {confidence_percentage:.1f}%")
        else:
            st.error(f"### 😞 Sentiment Outcome: NEGATIVE  |  Engine Confidence Score: {confidence_percentage:.1f}%")
        
                # Display Dynamic Charts Grid Layout
        st.markdown("### Metrics Analysis Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Overall Prediction Confidence")
            pos_pct = float(probabilities[0][1] * 100)
            neg_pct = float(probabilities[0][0] * 100)
            
            donut_data = pd.DataFrame({
                "Sentiment": ["Positive", "Negative"],
                "Percentage": [pos_pct, neg_pct]
            })
            
            fig_donut = px.pie(donut_data, names="Sentiment", values="Percentage", hole=0.5,
                               color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c"})
            fig_donut.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=300)
            st.plotly_chart(fig_donut, use_container_width=True)

        with col2:
            st.subheader("Words in Your Input")
            words_in_tweet = cleaned.split()
            word_counts = {}
            for word in words_in_tweet:
                if len(word) > 2:
                    word_counts[word] = word_counts.get(word, 0) + 1
            if word_counts:
                words_data = pd.DataFrame({
                    "Keywords": list(word_counts.keys()),
                    "Frequency": list(word_counts.values())
                }).sort_values(by="Frequency", ascending=False).head(10) # Show top 10 words max
                
                fig_bar = px.bar(words_data, x="Keywords", y="Frequency", color_discrete_sequence=["#1f77b4"])
                fig_bar.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=300)
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.write("Type a longer tweet to visualize key word frequencies.")

        st.markdown("### Analysis Details")
        details_df = pd.DataFrame({
            "Tweet Snippet String": [user_tweet],
            "Selected Pipeline Engine": [model_choice],
            "Classification Result Index": [int(prediction[0])],
            "Confidence Probability": [f"{confidence_percentage:.2f}%"]
        })
        st.dataframe(details_df, use_container_width=True, hide_index=True)
        
    else:
        st.warning("Please type a tweet before running the analysis engine.")
