import streamlit as st
import pandas as pd
import os
import gdown
from fastai.collab import load_learner
import plotly.express as px

# ------------------- Load Model -------------------
@st.cache_resource
def load_model():
    MODEL_URL = "https://drive.google.com/uc?id=1kzap_V1lrv7ihUk3dVD4waijxLRxHunJ"
    MODEL_PATH = "models/book_recommender_latest.pkl"
    os.makedirs("models", exist_ok=True)

    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 5_000_000:
        with st.spinner("🔽 Downloading model from Google Drive..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    with open(MODEL_PATH, 'rb') as f:
        if f.read(10).startswith(b'<html>'):
            st.error("❌ Model download failed. Got HTML instead of a .pkl file.")
            st.stop()

    return load_learner(MODEL_PATH)

# ------------------- Load Data -------------------
@st.cache_data
def load_ratings():
    return pd.read_csv("Ratings_cleaned.csv", dtype={"user_id": str, "book_id": str, "rating": int})

@st.cache_data
def load_books():
    return pd.read_csv("Books_cleaned.csv", dtype={"book_id": str})

@st.cache_data
def get_user_counts(ratings_df):
    df = ratings_df['user_id'].value_counts().reset_index()
    df.columns = ['user_id', 'num_ratings']
    return df.sort_values('num_ratings', ascending=False).head(100)

# ------------------- Streamlit UI -------------------
st.set_page_config("📚 Book Recommender", layout="wide")
st.title("📚 Personalized Book Recommender")

with st.spinner("🔄 Loading model and data..."):
    learn = load_model()
    ratings = load_ratings()
    books = load_books()
    user_counts = get_user_counts(ratings)

user_id = st.selectbox(
    "Select a user:",
    options=user_counts['user_id'],
    format_func=lambda x: f"User {x} ({user_counts[user_counts['user_id'] == x]['num_ratings'].values[0]} ratings)"
)

if st.button("🔍 Get Recommendations"):
    st.markdown("---")
    st.subheader(f"📖 Books You've Rated")
    user_rated = ratings[ratings['user_id'] == user_id].merge(books, on='book_id')
    st.dataframe(user_rated[['title', 'author', 'rating']].sort_values(by='rating', ascending=False), use_container_width=True)

    # 📊 Ratings Histogram
    st.subheader("📊 Rating Distribution")
    fig = px.histogram(user_rated, x="rating", nbins=10, title="Your Rating Distribution")
    st.plotly_chart(fig, use_container_width=True)

    seen_books = user_rated['book_id'].tolist()
    popular_books = ratings['book_id'].value_counts()[ratings['book_id'].value_counts() >= 5].index.tolist()
    candidate_books = books[~books['book_id'].isin(seen_books) & books['book_id'].isin(popular_books)]

    sample_books = candidate_books.sample(n=min(1000, len(candidate_books)), random_state=42)
    test_df = pd.DataFrame({
        "user_id": [user_id] * len(sample_books),
        "book_id": sample_books['book_id'].tolist()
    })

    with st.spinner("⚙️ Predicting ratings..."):
        preds = learn.get_preds(dl=learn.dls.test_dl(test_df))[0].squeeze().tolist()
        sample_books['predicted_rating'] = preds
        top_books = sample_books.sort_values(by='predicted_rating', ascending=False).head(10)

    st.subheader(f"📘 Top 10 Recommended Books for User {user_id}")
    st.dataframe(top_books[['title', 'author', 'predicted_rating']].reset_index(drop=True), use_container_width=True)

    st.subheader("📈 Prediction Distribution")
    fig2 = px.histogram(sample_books, x="predicted_rating", nbins=20, title="Predicted Ratings for Unseen Books")
    st.plotly_chart(fig2, use_container_width=True)
