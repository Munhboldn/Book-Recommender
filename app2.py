import streamlit as st
import pandas as pd
import os
import urllib.request
from fastai.collab import load_learner
import plotly.express as px

# --- Model Setup (downloads from Google Drive) ---
MODEL_URL = 'https://drive.google.com/uc?export=download&id=1kzap_V1lrv7ihUk3dVD4waijxLRxHunJ'
MODEL_PATH = 'models/book_recommender_latest.pkl'

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs('models', exist_ok=True)
        with st.spinner("🔽 Downloading model from Google Drive..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return load_learner(MODEL_PATH)

# --- Load Data ---
@st.cache_data
def load_ratings_data():
    path = 'Ratings_cleaned.csv'
    if not os.path.exists(path):
        st.error(f"❌ Ratings file missing: `{path}`")
        st.stop()
    return pd.read_csv(path, encoding='utf-8', dtype={'user_id': str, 'book_id': str, 'rating': int})

@st.cache_data
def load_books_data():
    path = 'Books_cleaned.csv'
    if not os.path.exists(path):
        st.error(f"❌ Books file missing: `{path}`")
        st.stop()
    df = pd.read_csv(path, encoding='utf-8', dtype={'book_id': str, 'year': int})
    required = ['book_id', 'title', 'author', 'year', 'publisher']
    if not all(col in df.columns for col in required):
        st.error("❌ `Books_cleaned.csv` is missing required columns.")
        st.stop()
    return df

# --- App Setup ---
st.set_page_config("📚 Book Recommender", layout="wide")
st.title("📚 Personalized Book Recommender")
st.markdown("Get AI-powered book recommendations based on user ratings.")
st.markdown("---")

with st.spinner("🔄 Loading model and data..."):
    learn = load_model()
    ratings = load_ratings_data()
    books = load_books_data()

# --- Sidebar Info ---
with st.sidebar:
    st.header("📦 Dataset Info")
    st.markdown(f"**Ratings:** {len(ratings):,}")
    st.markdown(f"**Books:** {books['book_id'].nunique():,}")
    st.markdown("Only books rated by ≥ 5 users are used for prediction.")
    st.markdown("---")
    st.caption("Built with FastAI + Streamlit")

# --- Popular books & top users ---
@st.cache_data
def get_popular_books(ratings_df):
    return ratings_df['book_id'].value_counts()[ratings_df['book_id'].value_counts() >= 5].index.tolist()

@st.cache_data
def get_top_users(ratings_df, n=100):
    user_counts = ratings_df['user_id'].value_counts().reset_index()
    user_counts.columns = ['user_id', 'num_ratings']
    return user_counts.sort_values('num_ratings', ascending=False).head(n)

popular_books = get_popular_books(ratings)
top_users_df = get_top_users(ratings)

# --- User Selection ---
col1, col2 = st.columns([3, 1])
with col1:
    user_id = st.selectbox(
        "Select a User (Top 100 most active)",
        options=top_users_df['user_id'].tolist(),
        format_func=lambda x: f"User {x}"
    )
with col2:
    num = top_users_df[top_users_df['user_id'] == user_id]['num_ratings'].values[0]
    st.metric("Ratings by User", f"{num}")

short_id = user_id[-4:] if len(user_id) > 4 else user_id

# --- Recommendations ---
if st.button("🔍 Get Recommendations"):
    st.subheader(f"📖 Books Rated by User #{short_id}")

    user_rated_ids = ratings[ratings['user_id'] == user_id]['book_id'].tolist()
    user_rated_df = ratings[ratings['user_id'] == user_id].merge(books, on='book_id')
    rated_view = user_rated_df[['title', 'author', 'rating']].copy()
    rated_view.columns = ['Title', 'Author', 'Your Rating']
    rated_view = rated_view.sort_values('Your Rating', ascending=False)

    if not rated_view.empty:
        st.dataframe(rated_view.reset_index(drop=True), use_container_width=True)

        st.markdown("### 📊 How You Usually Rate Books")
        st.caption("This chart shows how often you gave each rating (1 = worst, 10 = best)")
        fig1 = px.histogram(
            rated_view,
            x='Your Rating',
            nbins=10,
            title="How Often You Gave Each Rating",
            labels={"Your Rating": "Rating You Gave", "count": "Number of Books"}
        )
        fig1.update_layout(
            xaxis_title="Rating You Gave (1–10)",
            yaxis_title="Number of Books",
            bargap=0.1
        )
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("This user has not rated any books.")

    st.markdown("---")
    st.subheader(f"📘 Top 10 Recommended Books for User #{short_id}")

    unseen_df = books[
        (~books['book_id'].isin(user_rated_ids)) &
        (books['book_id'].isin(popular_books))
    ].copy()

    if unseen_df.empty:
        st.warning("This user has rated all popular books.")
    else:
        unseen_df = unseen_df.sample(min(1000, len(unseen_df)), random_state=42)
        test_df = pd.DataFrame({'user_id': [user_id] * len(unseen_df), 'book_id': unseen_df['book_id'].tolist()})

        with st.spinner("⚙️ Predicting ratings..."):
            try:
                preds = learn.get_preds(dl=learn.dls.test_dl(test_df))[0].squeeze().tolist()
                unseen_df['predicted_rating'] = preds
                top_books = unseen_df.sort_values('predicted_rating', ascending=False).head(10)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                top_books = pd.DataFrame()

        if not top_books.empty:
            top_view = top_books[['title', 'author', 'predicted_rating']].copy()
            top_view.columns = ['Title', 'Author', 'Predicted Rating']
            top_view['Predicted Rating'] = top_view['Predicted Rating'].round(2)
            st.dataframe(top_view.reset_index(drop=True), use_container_width=True)

            st.markdown("### 📊 How the AI Thinks You'll Rate New Books")
            st.caption("This chart shows the model’s guesses for how you’ll rate books you haven’t read yet")
            fig2 = px.histogram(
                unseen_df,
                x='predicted_rating',
                nbins=20,
                title="Predicted Ratings for Books You Haven't Seen Yet",
                labels={"predicted_rating": "AI-Predicted Rating", "count": "Number of Books"}
            )
            fig2.update_layout(
                xaxis_title="AI-Predicted Rating (1–10)",
                yaxis_title="Number of Books",
                bargap=0.1
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("No recommendations could be generated.")

# --- Reset Button ---
if st.button("🔄 Reset User Selection"):
    st.experimental_rerun()
