import streamlit as st
import pandas as pd
import os
import gdown
from fastai.collab import load_learner
import plotly.express as px

MODEL_URL = "https://drive.google.com/uc?id=1kzap_V1lrv7ihUk3dVD4waijxLRxHunJ"
MODEL_PATH = "models/book_recommender_latest.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        with st.spinner("📥 Downloading model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return load_learner(MODEL_PATH)

@st.cache_data
def load_ratings_data():
    return pd.read_csv("Ratings_cleaned.csv", encoding="utf-8", dtype={"user_id": str, "book_id": str, "rating": int})

@st.cache_data
def load_books_data():
    return pd.read_csv("Books_cleaned.csv", encoding="utf-8", dtype={"book_id": str, "year": int})

st.set_page_config("📚 Book Recommender", layout="wide")
st.title("📚 Personalized Book Recommendations")

with st.spinner("🔄 Loading model and data..."):
    learn = load_model()
    ratings = load_ratings_data()
    books = load_books_data()

user_counts = ratings["user_id"].value_counts().reset_index()
user_counts.columns = ["user_id", "num_ratings"]
top_users = user_counts.head(100)

user_id = st.selectbox("Select a user", options=top_users["user_id"])

if st.button("🔍 Get Recommendations"):
    user_rated = ratings[ratings["user_id"] == user_id]
    seen_books = user_rated["book_id"].tolist()

    user_rated_details = user_rated.merge(books, on="book_id")[["title", "author", "rating"]]
    st.subheader("📖 Books You've Rated")
    st.dataframe(user_rated_details.reset_index(drop=True), use_container_width=True)

    with st.expander("📊 Rating Distribution"):
        fig = px.histogram(user_rated, x="rating", nbins=10, title="Distribution of Your Ratings")
        st.plotly_chart(fig, use_container_width=True)

    popular_books = ratings["book_id"].value_counts()[ratings["book_id"].value_counts() >= 5].index
    unseen_books = books[~books["book_id"].isin(seen_books) & books["book_id"].isin(popular_books)]

    sample = unseen_books.sample(min(1000, len(unseen_books)), random_state=42)
    test_df = pd.DataFrame({"user_id": [user_id] * len(sample), "book_id": sample["book_id"].tolist()})

    with st.spinner("⚙️ Predicting..."):
        preds = learn.get_preds(dl=learn.dls.test_dl(test_df))[0].squeeze().tolist()
        sample["predicted_rating"] = preds
        top = sample.sort_values("predicted_rating", ascending=False).head(10)

    st.subheader("📘 Top 10 Recommended Books for this user")
    st.dataframe(top[["title", "author", "predicted_rating"]].reset_index(drop=True), use_container_width=True)

    with st.expander("📊 Prediction Distribution"):
        fig = px.histogram(sample, x="predicted_rating", nbins=15, title="Distribution of Predicted Ratings")
        st.plotly_chart(fig, use_container_width=True)
