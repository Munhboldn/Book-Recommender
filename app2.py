import streamlit as st
import pandas as pd
import os
import gdown
from fastai.collab import load_learner
import plotly.express as px

# --- Model Loading ---
MODEL_URL = "https://drive.google.com/uc?id=1kzap_V1lrv7ihUk3dVD4waijxLRxHunJ&export=download"
MODEL_PATH = "models/book_recommender_latest.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        with st.spinner("🔽 Downloading model from Google Drive..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return load_learner(MODEL_PATH)

# --- Load Data ---
@st.cache_data
def load_ratings_data():
    path = 'Ratings_cleaned.csv'
    if not os.path.exists(path):
        st.error("❌ Ratings file not found. Please upload or generate it.")
        st.stop()
    return pd.read_csv(path, dtype={'user_id': str, 'book_id': str, 'rating': int})

@st.cache_data
def load_books_data():
    path = 'Books_cleaned.csv'
    if not os.path.exists(path):
        st.error("❌ Books file not found. Please upload or generate it.")
        st.stop()
    return pd.read_csv(path, dtype={'book_id': str})

# --- Preprocessing ---
@st.cache_data
def get_popular_books(ratings_df):
    return ratings_df['book_id'].value_counts()[ratings_df['book_id'].value_counts() >= 5].index.tolist()

@st.cache_data
def get_top_users(ratings_df, n=100):
    user_counts = ratings_df['user_id'].value_counts().reset_index()
    user_counts.columns = ['user_id', 'num_ratings']
    return user_counts.sort_values('num_ratings', ascending=False).head(n)

# --- App UI ---
st.set_page_config("📚 Book Recommender", layout="wide")
st.title("📚 Personalized Book Recommender")
st.caption("Powered by FastAI & Streamlit")

# Load model and data
with st.spinner("🔄 Loading model and data..."):
    learn = load_model()
    ratings = load_ratings_data()
    books = load_books_data()

# Pre-calculate
popular_books = get_popular_books(ratings)
top_users_df = get_top_users(ratings)

# --- User selection ---
col1, col2 = st.columns([3, 1])
with col1:
    user_id = st.selectbox("Select a User (Top 100 active)", top_users_df['user_id'].tolist())
with col2:
    user_ratings_count = top_users_df[top_users_df['user_id'] == user_id]['num_ratings'].values[0]
    st.metric("User Ratings", f"{user_ratings_count}")

# --- Get Recommendations ---
if st.button("🔍 Get Recommendations"):
    # Rated books
    user_rated = ratings[ratings['user_id'] == user_id].merge(books, on='book_id')
    seen_ids = user_rated['book_id'].tolist()

    st.subheader("📖 Books You've Rated")
    if not user_rated.empty:
        display_rated = user_rated[['title', 'author', 'rating']].sort_values('rating', ascending=False)
        st.dataframe(display_rated.rename(columns={
            'title': 'Title', 'author': 'Author', 'rating': 'Your Rating'
        }).reset_index(drop=True), use_container_width=True)

        # Plot rating histogram
        fig1 = px.histogram(display_rated, x='rating', nbins=10, title="How You Rate Books")
        fig1.update_layout(xaxis_title="Rating", yaxis_title="Count", bargap=0.1)
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("This user has not rated any books.")

    # Unseen books
    unseen_books = books[
        (~books['book_id'].isin(seen_ids)) &
        (books['book_id'].isin(popular_books))
    ].copy()

    if unseen_books.empty:
        st.warning("No unseen popular books to recommend.")
    else:
        st.subheader("📘 Top 10 Recommended Books for You")
        sample = unseen_books.sample(min(1000, len(unseen_books)), random_state=42)
        test_df = pd.DataFrame({'user_id': [user_id]*len(sample), 'book_id': sample['book_id']})
        with st.spinner("⚙️ Predicting..."):
            try:
                preds = learn.get_preds(dl=learn.dls.test_dl(test_df))[0].squeeze().tolist()
                sample['predicted_rating'] = preds
                top_recs = sample.sort_values('predicted_rating', ascending=False).head(10)
                top_display = top_recs[['title', 'author', 'predicted_rating']]
                top_display.columns = ['Title', 'Author', 'Predicted Rating']
                top_display['Predicted Rating'] = top_display['Predicted Rating'].round(2)
                st.dataframe(top_display.reset_index(drop=True), use_container_width=True)

                # Plot prediction distribution
                fig2 = px.histogram(sample, x='predicted_rating', nbins=20, title="Predicted Ratings Distribution")
                fig2.update_layout(xaxis_title="Predicted Rating", yaxis_title="Books", bargap=0.1)
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
