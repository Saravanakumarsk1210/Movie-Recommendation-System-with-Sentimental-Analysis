import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from scipy.special import softmax
import random
import requests
import torch

# Page configuration
st.set_page_config(
    page_title="🎬📚 Sentiment-Based Movie & Book Recommender",
    page_icon="🎭",
    layout="wide"
)

# Load BERT model for sentiment analysis
@st.cache_resource
def load_sentiment_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Load genre classification model
@st.cache_resource
def load_genre_model():
    try:
        # Using a zero-shot classification pipeline for genre detection
        genre_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        return genre_classifier
    except Exception as e:
        st.error(f"Error loading genre model: {e}")
        return None

# Load datasets
@st.cache_data
def load_datasets():
    try:
        # Load movie dataset
        movies_df = pd.read_csv('top10K-TMDB-movies.csv')
        movies_df = movies_df[['id', 'title', 'genre', 'overview', 'popularity', 'vote_average']].copy()
        
        # Load book dataset
        books_df = pd.read_csv('book.csv')
        books_df = books_df[['title', 'name', 'genre', 'rating', 'synopsis']].copy()
        
        return movies_df, books_df
    except FileNotFoundError as e:
        st.error(f"Dataset not found: {e}")
        return None, None

# Sentiment analysis function
def analyze_sentiment(text, tokenizer, model):
    # Preprocess text (handle mentions and links)
    tweet_words = []
    for word in text.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)
    
    tweet_proc = " ".join(tweet_words)
    
    # Encode and predict
    encoded_tweet = tokenizer(tweet_proc, return_tensors='pt', max_length=512, truncation=True)
    output = model(**encoded_tweet)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    
    # Map to sentiment labels
    labels = ['Negative', 'Neutral', 'Positive']
    sentiment_scores = {labels[i]: float(scores[i]) for i in range(len(labels))}
    predicted_sentiment = labels[scores.argmax()]
    
    return predicted_sentiment, sentiment_scores

# Genre detection function
def detect_genre(text, genre_classifier):
    if genre_classifier is None:
        return "Unknown", {}
    
    # Define comprehensive genre labels for movies and books
    genre_labels = [
        "Action", "Adventure", "Animation", "Biography", "Comedy", "Crime", "Documentary", 
        "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery", "Romance", 
        "Science Fiction", "Thriller", "War", "Western", "Psychological", "Philosophical",
        "Self-help", "Educational", "Literary Fiction", "Young Adult", "Children",
        "Memoir", "True Crime", "Health", "Business", "Technology", "Travel", "Cooking"
    ]
    
    try:
        # Use zero-shot classification to detect genre
        result = genre_classifier(text, genre_labels)
        
        # Get top genre and confidence scores
        predicted_genre = result['labels'][0]
        genre_scores = {label: score for label, score in zip(result['labels'], result['scores'])}
        
        # Return top 3 genres with scores
        top_genres = {label: score for label, score in list(genre_scores.items())[:3]}
        
        return predicted_genre, top_genres
        
    except Exception as e:
        st.error(f"Error in genre detection: {e}")
        return "Unknown", {}

# Function to fetch movie poster
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=c7ec19ffdd3279641fb606d19ceb9bb1&language=en-US"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if 'poster_path' in data and data['poster_path']:
            poster_path = data['poster_path']
            full_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
            return full_path
        else:
            return "https://via.placeholder.com/500x750?text=No+Image"
    except:
        return "https://via.placeholder.com/500x750?text=No+Image"

# Recommend movies based on sentiment and genre
def recommend_movies_by_sentiment_and_genre(sentiment, detected_genre, movies_df, num_recommendations=5):
    # First try to filter by detected genre
    genre_filtered = movies_df[movies_df['genre'].str.contains(detected_genre, na=False, case=False)]
    
    if len(genre_filtered) >= num_recommendations:
        filtered_movies = genre_filtered
    else:
        # Fallback to sentiment-based filtering
        if sentiment == 'Positive':
            filtered_movies = movies_df[movies_df['vote_average'] >= 7.5].copy()
            preferred_genres = ['Comedy', 'Family', 'Romance', 'Animation']
            for genre in preferred_genres:
                genre_movies = filtered_movies[filtered_movies['genre'].str.contains(genre, na=False, case=False)]
                if len(genre_movies) >= num_recommendations:
                    filtered_movies = genre_movies
                    break
        elif sentiment == 'Negative':
            filtered_movies = movies_df.copy()
            preferred_genres = ['Drama', 'Thriller', 'Horror', 'Crime', 'War']
            for genre in preferred_genres:
                genre_movies = filtered_movies[filtered_movies['genre'].str.contains(genre, na=False, case=False)]
                if len(genre_movies) >= num_recommendations:
                    filtered_movies = genre_movies
                    break
        else:  # Neutral
            filtered_movies = movies_df.copy()
            preferred_genres = ['Sci-Fi', 'Adventure', 'Action', 'Mystery']
            for genre in preferred_genres:
                genre_movies = filtered_movies[filtered_movies['genre'].str.contains(genre, na=False, case=False)]
                if len(genre_movies) >= num_recommendations:
                    filtered_movies = genre_movies
                    break
    
    # Sort by popularity and rating, then sample
    if not filtered_movies.empty:
        filtered_movies = filtered_movies.sort_values(['popularity', 'vote_average'], ascending=[False, False])
        sample_size = min(num_recommendations, len(filtered_movies))
        recommended_movies = filtered_movies.head(sample_size * 2).sample(n=sample_size)
        return recommended_movies
    else:
        # Fallback to top popular movies
        return movies_df.nlargest(num_recommendations, 'popularity')

# Recommend books based on sentiment and genre
def recommend_books_by_sentiment_and_genre(sentiment, detected_genre, books_df, num_recommendations=5):
    # First try to filter by detected genre
    genre_filtered = books_df[books_df['genre'].str.contains(detected_genre, na=False, case=False)]
    
    if len(genre_filtered) >= num_recommendations:
        filtered_books = genre_filtered
    else:
        # Fallback to sentiment-based filtering
        if sentiment == 'Positive':
            filtered_books = books_df[books_df['rating'] >= 4.0].copy()
            avoid_genres = ['horror', 'thriller', 'mystery']
            for genre in avoid_genres:
                filtered_books = filtered_books[~filtered_books['genre'].str.contains(genre, na=False, case=False)]
        elif sentiment == 'Negative':
            filtered_books = books_df.copy()
            preferred_genres = ['history', 'biography', 'psychology', 'philosophy']
            for genre in preferred_genres:
                genre_books = filtered_books[filtered_books['genre'].str.contains(genre, na=False, case=False)]
                if len(genre_books) >= num_recommendations:
                    filtered_books = genre_books
                    break
        else:  # Neutral
            filtered_books = books_df.copy()
    
    # Sort by rating and sample
    if not filtered_books.empty:
        filtered_books = filtered_books.sort_values('rating', ascending=False)
        sample_size = min(num_recommendations, len(filtered_books))
        recommended_books = filtered_books.head(sample_size * 2).sample(n=sample_size)
        return recommended_books
    else:
        # Fallback to top rated books
        return books_df.nlargest(num_recommendations, 'rating')

# Main app
def main():
    st.title("🎬📚 Sentiment & Genre-Based Movie & Book Recommender")
    st.markdown("**Discover movies and books tailored to your mood and preferred genre!**")
    
    # Load models with loading indicator
    with st.spinner("Loading AI models... Please wait"):
        try:
            # Load sentiment model
            sentiment_tokenizer, sentiment_model = load_sentiment_model()
            # Load genre model
            genre_classifier = load_genre_model()
            st.success("✅ Models loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            return
    
    # Load datasets
    with st.spinner("Loading movie and book datasets..."):
        movies_df, books_df = load_datasets()
        if movies_df is None or books_df is None:
            st.error("❌ Could not load datasets. Please check file paths.")
            return
        st.success(f"✅ Loaded {len(movies_df)} movies and {len(books_df)} books!")
    
    # User input
    st.subheader("📝 Share your thoughts:")
    user_input = st.text_area(
        "Tell me how you're feeling or what's on your mind...",
        placeholder="e.g., I'm feeling great today and want something romantic! or I'm having a tough time and need something thrilling...",
        height=100
    )
    
    # Analysis and recommendations
    if st.button("🔍 Analyze & Recommend", type="primary"):
        if user_input.strip():
            # Perform sentiment analysis
            with st.spinner("Analyzing your sentiment..."):
                predicted_sentiment, sentiment_scores = analyze_sentiment(user_input, sentiment_tokenizer, sentiment_model)
            
            # Perform genre detection
            with st.spinner("Detecting preferred genre..."):
                predicted_genre, genre_scores = detect_genre(user_input, genre_classifier)
            
            # Display analysis results
            st.subheader("🎭 AI Analysis Results")
            
            # Create columns for results display
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 😊 Sentiment Analysis")
                # Display dominant sentiment
                sentiment_emoji = {"Positive": "😊", "Neutral": "😐", "Negative": "😔"}
                st.metric(
                    label="Detected Sentiment",
                    value=f"{sentiment_emoji.get(predicted_sentiment, '🤔')} {predicted_sentiment}",
                    delta=f"Confidence: {sentiment_scores[predicted_sentiment]:.1%}"
                )
                
                # Display sentiment probabilities
                st.write("**Sentiment Probabilities:**")
                for sentiment, score in sentiment_scores.items():
                    st.progress(score, text=f"{sentiment}: {score:.1%}")
            
            with col2:
                st.markdown("### 🎭 Genre Detection")
                # Display dominant genre
                genre_emoji = {
                    "Action": "💥", "Adventure": "🗺️", "Animation": "🎨", "Biography": "📖", 
                    "Comedy": "😂", "Crime": "🔍", "Documentary": "🎥", "Drama": "🎭", 
                    "Family": "👨‍👩‍👧‍👦", "Fantasy": "🧙‍♂️", "History": "📜", "Horror": "👻", 
                    "Music": "🎵", "Mystery": "🔍", "Romance": "💕", "Science Fiction": "🚀", 
                    "Thriller": "😱", "War": "⚔️", "Western": "🤠"
                }
                st.metric(
                    label="Detected Genre",
                    value=f"{genre_emoji.get(predicted_genre, '🎬')} {predicted_genre}",
                    delta=f"Confidence: {list(genre_scores.values())[0]:.1%}" if genre_scores else "N/A"
                )
                
                # Display top genre probabilities
                if genre_scores:
                    st.write("**Top Genre Predictions:**")
                    for genre, score in list(genre_scores.items())[:3]:
                        st.progress(score, text=f"{genre}: {score:.1%}")
            
            st.divider()
            
            # Get recommendations
            with st.spinner("Finding perfect recommendations for you..."):
                recommended_movies = recommend_movies_by_sentiment_and_genre(
                    predicted_sentiment, predicted_genre, movies_df, 5
                )
                recommended_books = recommend_books_by_sentiment_and_genre(
                    predicted_sentiment, predicted_genre, books_df, 5
                )
            
            # Display movie recommendations
            st.subheader(f"🎬 Recommended Movies (Based on {predicted_sentiment} mood & {predicted_genre} preference)")
            if not recommended_movies.empty:
                movie_cols = st.columns(5)
                for idx, (_, movie) in enumerate(recommended_movies.iterrows()):
                    with movie_cols[idx % 5]:
                        st.write(f"**{movie['title'][:30]}{'...' if len(movie['title']) > 30 else ''}**")
                        poster_url = fetch_poster(movie['id'])
                        st.image(poster_url, width=120)
                        st.write(f"⭐ {movie['vote_average']:.1f}")
                        st.write(f"🎭 {movie['genre'][:20]}{'...' if len(str(movie['genre'])) > 20 else ''}")
                        
                        # Show overview in expander
                        with st.expander("Plot"):
                            st.write(movie['overview'][:200] + "..." if len(str(movie['overview'])) > 200 else movie['overview'])
            else:
                st.info("No movie recommendations found.")
            
            st.divider()
            
            # Display book recommendations
            st.subheader(f"📚 Recommended Books (Based on {predicted_sentiment} mood & {predicted_genre} preference)")
            if not recommended_books.empty:
                for _, book in recommended_books.iterrows():
                    with st.container():
                        book_col1, book_col2 = st.columns([3, 1])
                        
                        with book_col1:
                            st.write(f"**📖 {book['title']}**")
                            st.write(f"*by {book['name']}*")
                            if pd.notna(book['synopsis']):
                                synopsis = str(book['synopsis'])
                                st.write(f"_{synopsis[:200]}{'...' if len(synopsis) > 200 else ''}_")
                        
                        with book_col2:
                            st.metric("Rating", f"⭐ {book['rating']:.1f}")
                            st.write(f"📚 {book['genre'].title()}")
                        
                        st.divider()
            else:
                st.info("No book recommendations found.")
            
        else:
            st.warning("⚠️ Please enter some text to analyze.")
    
    # Sidebar with info
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("This app uses AI to analyze both sentiment and genre preferences from your text to recommend movies and books.")
        
        st.header("📊 Dataset Info")
        if 'movies_df' in locals():
            st.write(f"🎬 Movies: {len(movies_df):,}")
            st.write(f"📚 Books: {len(books_df):,}")
        
        st.header("🤖 How it Works")
        st.write("1. **Enter** your thoughts or feelings")
        st.write("2. **AI analyzes** sentiment (Positive/Neutral/Negative)")
        st.write("3. **AI detects** preferred genre from your text")
        st.write("4. **Get personalized** movie & book recommendations")
        
        st.header("🎭 Supported Genres")
        st.write("Action, Adventure, Animation, Biography, Comedy, Crime, Drama, Family, Fantasy, History, Horror, Mystery, Romance, Sci-Fi, Thriller, and more!")
        
        st.info("💡 **Tip:** Be descriptive about both your mood AND genre preferences for better recommendations!")

if __name__ == "__main__":
    main()
