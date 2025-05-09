import streamlit as st
import pandas as pd
import os
import gdown
import pathlib
import sys
from fastai.collab import load_learner
import plotly.express as px

MODEL_URL = 'https://drive.google.com/uc?id=1kzap_V1lrv7ihUk3dVD4waijxLRxHunJ'
MODEL_PATH = 'models/book_recommender_latest.pkl'

if sys.platform != "win32":
    pathlib.WindowsPath = pathlib.PosixPath

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        with st.spinner("📥 Downloading model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return load_learner(MODEL_PATH)

@st.cache_data
def load_ratings_data():
    return pd.read_csv("Ratings_cleaned.csv", dtype={'user_id': str, 'book_id': str, 'rating': int})

@st.cache_data
def load_books_data():
    return pd.read_csv("Books_cleaned.csv", dtype={'book_id': str})

@st.cache_data
def get_user_counts(ratings):
    user_counts_df = ratings['user_id'].value_counts().reset_index()
    user_counts_df.columns = ['user_id', 'num_ratings']
    return user_counts_df.sort_values('num_ratings', ascending=False).head(100)

@st.cache_data
def get_popular_books(ratings):
    return ratings['book_id'].value_counts()[ratings['book_id'].value_counts() >= 5].index.tolist()

st.set_page_config("📚 Book Recommender", layout="wide")
st.title(":blue_book: Personalized Book Recommendations")
st.markdown("---")

with st.spinner("🔄 Loading model and data..."):
    learn = load_model()
    ratings = load_ratings_data()
    books = load_books_data()

user_counts = get_user_counts(ratings)
popular_books = get_popular_books(ratings)

user_id = st.selectbox(
    "Select a User",
    options=user_counts['user_id'].tolist(),
    format_func=lambda x: f"User {x} — {user_counts[user_counts['user_id'] == x]['num_ratings'].values[0]} ratings"
)

if st.button("🔍 Get Recommendations"):
    st.subheader(f"📖 Books You've Rated")
    user_rated_ids = ratings[ratings['user_id'] == user_id]['book_id'].tolist()
    rated_books = ratings[ratings['user_id'] == user_id].merge(books, on='book_id')
    rated_books = rated_books[['title', 'author', 'rating']].sort_values('rating', ascending=False)
    with st.expander("See All Rated Books"):
        st.dataframe(rated_books, use_container_width=True)

    fig = px.histogram(rated_books, x='rating', nbins=10, title="Distribution of Your Ratings")
    st.plotly_chart(fig, use_container_width=True)

    unseen_books = books[~books['book_id'].isin(user_rated_ids) & books['book_id'].isin(popular_books)].copy()
    if unseen_books.empty:
        st.warning("No unseen popular books available.")
    else:
        test_df = pd.DataFrame({'user_id': [user_id]*len(unseen_books), 'book_id': unseen_books['book_id'].tolist()})
        with st.spinner("Predicting ratings..."):
            preds = learn.get_preds(dl=learn.dls.test_dl(test_df))[0].squeeze().tolist()
            unseen_books['predicted_rating'] = preds
            fig_pred = px.histogram(unseen_books, x='predicted_rating', nbins=20, title="Distribution of Predicted Ratings")
            st.plotly_chart(fig_pred, use_container_width=True)
            top_books = unseen_books.sort_values('predicted_rating', ascending=False).head(10)

        st.subheader(f"📘 Top 10 Recommended Books for User {user_id}")
        st.dataframe(top_books[['title', 'author', 'predicted_rating']], use_container_width=True)
