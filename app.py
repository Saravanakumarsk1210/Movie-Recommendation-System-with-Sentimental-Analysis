# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sentiment import SentimentAnalyzer, MediaRecommender
import time
import random
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud

# Set page config
st.set_page_config(
    page_title="Movie and Book Recommendations with Genre Mapping and Twitter Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize our classes
@st.cache_resource
def load_models():
    sentiment_analyzer = SentimentAnalyzer()
    media_recommender = MediaRecommender()
    return sentiment_analyzer, media_recommender

sentiment_analyzer, media_recommender = load_models()

# Custom CSS for styling
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
    }
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: #1E90FF;
        margin-bottom: 1rem;
    }
    .section-heading {
        font-size: 1.8rem;
        font-weight: 600;
        color: #333;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #f0f0f0;
    }
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background-color: white;
        margin-bottom: 1rem;
    }
    .media-title {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .sentiment-positive {
        color: #2ecc71;
        font-weight: 600;
    }
    .sentiment-neutral {
        color: #3498db;
        font-weight: 600;
    }
    .sentiment-negative {
        color: #e74c3c;
        font-weight: 600;
    }
    .divider {
        margin: 2rem 0;
        border-top: 1px solid #f0f0f0;
    }
    .result-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1E90FF;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # App title and description
    st.markdown('<p class="main-title">Movie and Book Recommendations with Genre Mapping and Twitter Sentiment Analysis</p>', unsafe_allow_html=True)
    
    st.markdown("""
    This application analyzes the sentiment of tweets or text input and recommends movies and books 
    based on the detected emotional tone. The recommendation system maps sentiment to relevant genres 
    and provides personalized media suggestions.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This project combines natural language processing and recommendation systems to:
        
        1. Analyze sentiment using a RoBERTa model trained on tweets
        2. Map sentiment profiles to relevant genres
        3. Recommend books and movies based on sentiment
        
        Ideal for discovering content that matches your mood or helping others find media aligned with their emotional state.
        """)
        
        st.header("Options")
        rec_count = st.slider("Number of recommendations", 3, 10, 5)
        show_genre_mapping = st.checkbox("Show genre mapping details", value=False)
        
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Single Text Analysis", "Batch Analysis (CSV)", "Explore Genre Mapping"])
    
    # Single Text Analysis Tab
    with tab1:
        st.markdown('<p class="section-heading">Single Text Analysis</p>', unsafe_allow_html=True)
        
        st.markdown("""
        Enter a tweet or text to analyze its sentiment and get personalized book and movie recommendations.
        Try expressing a specific mood or emotion in your text to see how it affects recommendations.
        """)
        
        user_text = st.text_area("Enter your text:", height=120, 
                             placeholder="Enter any text, tweet, or statement describing your current mood or thoughts...")
        
        col1, col2 = st.columns(2)
        with col1:
            analyze_button = st.button("📊 Analyze Sentiment & Get Recommendations", type="primary", use_container_width=True)
        with col2:
            example_button = st.button("🎲 Try a Random Example", use_container_width=True)
        
        if example_button:
            examples = [
                "Just watched an amazing sunset at the beach! Life is beautiful! #blessed #grateful",
                "Feeling so frustrated with this terrible customer service. Been on hold for 45 minutes! 😡",
                "Interesting documentary about climate change. Makes you think about our future.",
                # app.py (continued)
                "My new job is amazing! The team is so supportive and I'm learning so much every day!",
                "Just finished reading a depressing book. Can't shake off the heavy feeling.",
                "Not sure how I feel about the new restaurant that opened nearby. Food was okay, I guess."
            ]
            user_text = random.choice(examples)
            st.session_state.user_text = user_text
            time.sleep(0.5)  # Give the UI time to update
            analyze_button = True  # Trigger analysis
        
        if analyze_button and user_text:
            with st.spinner("Analyzing sentiment..."):
                # Get sentiment analysis results
                sentiment_result = sentiment_analyzer.analyze_sentiment(user_text)
                
                # Store results in session state
                st.session_state.sentiment_result = sentiment_result
                st.session_state.user_text = user_text
                
                # Display sentiment analysis results
                display_sentiment_results(sentiment_result, user_text)
                
                # Get and display recommendations based on sentiment probabilities
                display_media_recommendations(sentiment_result, rec_count)
                
                # Display genre mapping if selected
                if show_genre_mapping:
                    display_genre_mapping(sentiment_result["max_sentiment"])
        
        # If text was analyzed previously, keep showing results
        elif hasattr(st.session_state, 'sentiment_result') and hasattr(st.session_state, 'user_text'):
            display_sentiment_results(st.session_state.sentiment_result, st.session_state.user_text)
            display_media_recommendations(st.session_state.sentiment_result, rec_count)
            if show_genre_mapping:
                display_genre_mapping(st.session_state.sentiment_result["max_sentiment"])
    
    # Batch Analysis Tab
    with tab2:
        st.markdown('<p class="section-heading">Batch Analysis from CSV</p>', unsafe_allow_html=True)
        st.markdown("""
        Upload a CSV file with tweets or text entries for batch sentiment analysis.
        The file should have a header row with column names.
        """)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        
        if uploaded_file is not None:
            # Load the CSV file
            try:
                df = pd.read_csv(uploaded_file)
                
                # Show dataframe preview
                st.subheader("Preview of uploaded data")
                st.dataframe(df.head())
                
                # Let user select which column contains the text to analyze
                text_column = st.selectbox("Select the column containing text to analyze:", df.columns)
                
                if st.button("Run Batch Analysis"):
                    with st.spinner("Analyzing sentiments for all texts..."):
                        # Process each text in the dataframe
                        results = []
                        progress_bar = st.progress(0)
                        
                        for i, text in enumerate(df[text_column]):
                            if isinstance(text, str):  # Make sure the text is a string
                                sentiment_result = sentiment_analyzer.analyze_sentiment(text)
                                results.append({
                                    "text": text[:100] + "..." if len(text) > 100 else text,
                                    "max_sentiment": sentiment_result["max_sentiment"],
                                    "positive": sentiment_result["probabilities"]["Positive"],
                                    "neutral": sentiment_result["probabilities"]["Neutral"],
                                    "negative": sentiment_result["probabilities"]["Negative"]
                                })
                            else:
                                results.append({
                                    "text": "Invalid text",
                                    "max_sentiment": "N/A",
                                    "positive": 0,
                                    "neutral": 0,
                                    "negative": 0
                                })
                            
                            # Update progress bar
                            progress_bar.progress((i + 1) / len(df))
                        
                        # Create a dataframe from results
                        results_df = pd.DataFrame(results)
                        
                        # Display the results
                        st.subheader("Sentiment Analysis Results")
                        st.dataframe(results_df)
                        
                        # Create visualization of sentiment distribution
                        st.subheader("Sentiment Distribution")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            create_sentiment_pie_chart(results_df)
                        
                        with col2:
                            create_sentiment_bar_chart(results_df)
                        
                        # Word cloud analysis
                        st.subheader("Word Frequency Analysis by Sentiment")
                        create_sentiment_word_clouds(df[text_column], results_df)
                        
                        # Allow downloading the results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="sentiment_analysis_results.csv",
                            mime="text/csv",
                        )
            
            except Exception as e:
                st.error(f"Error: {e}")
                st.error("Make sure your CSV file is properly formatted.")
    
    # Explore Genre Mapping Tab
    with tab3:
        st.markdown('<p class="section-heading">Explore Genre Mapping</p>', unsafe_allow_html=True)
        st.markdown("""
        This section shows how different sentiments are mapped to genres for books and movies.
        Understanding these mappings helps explain why certain titles are recommended based on sentiment analysis.
        """)
        
        # Create columns for each sentiment
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<h3 class="sentiment-positive">Positive Sentiment</h3>', unsafe_allow_html=True)
            st.markdown("#### Associated Genres:")
            for genre in media_recommender.genre_sentiment_mapping["Positive"]:
                st.markdown(f"- {genre}")
            
            st.markdown("#### Sample Books:")
            for book in media_recommender.books["Positive"][:3]:
                st.markdown(f"- **{book['title']}** by {book['author']} ({book['genre']})")
            
            st.markdown("#### Sample Movies:")
            for movie in media_recommender.movies["Positive"][:3]:
                st.markdown(f"- **{movie['title']}** ({movie['release_year']}) - {movie['genre']}")
        
        with col2:
            st.markdown('<h3 class="sentiment-neutral">Neutral Sentiment</h3>', unsafe_allow_html=True)
            st.markdown("#### Associated Genres:")
            for genre in media_recommender.genre_sentiment_mapping["Neutral"]:
                st.markdown(f"- {genre}")
            
            st.markdown("#### Sample Books:")
            for book in media_recommender.books["Neutral"][:3]:
                st.markdown(f"- **{book['title']}** by {book['author']} ({book['genre']})")
            
            st.markdown("#### Sample Movies:")
            for movie in media_recommender.movies["Neutral"][:3]:
                st.markdown(f"- **{movie['title']}** ({movie['release_year']}) - {movie['genre']}")
        
        with col3:
            st.markdown('<h3 class="sentiment-negative">Negative Sentiment</h3>', unsafe_allow_html=True)
            st.markdown("#### Associated Genres:")
            for genre in media_recommender.genre_sentiment_mapping["Negative"]:
                st.markdown(f"- {genre}")
            
            st.markdown("#### Sample Books:")
            for book in media_recommender.books["Negative"][:3]:
                st.markdown(f"- **{book['title']}** by {book['author']} ({book['genre']})")
            
            st.markdown("#### Sample Movies:")
            for movie in media_recommender.movies["Negative"][:3]:
                st.markdown(f"- **{movie['title']}** ({movie['release_year']}) - {movie['genre']}")
        
        # Show genre frequency distribution
        st.markdown('<p class="section-heading">Genre Distribution Across Sentiments</p>', unsafe_allow_html=True)
        
        # Extract all genres from books and movies
        genre_counts = {"Positive": {}, "Neutral": {}, "Negative": {}}
        
        for sentiment in ["Positive", "Neutral", "Negative"]:
            # Process books
            for book in media_recommender.books[sentiment]:
                genres = [g.strip() for g in book["genre"].split(",")]
                for genre in genres:
                    if genre in genre_counts[sentiment]:
                        genre_counts[sentiment][genre] += 1
                    else:
                        genre_counts[sentiment][genre] = 1
            
            # Process movies
            for movie in media_recommender.movies[sentiment]:
                genres = [g.strip() for g in movie["genre"].split(",")]
                for genre in genres:
                    if genre in genre_counts[sentiment]:
                        genre_counts[sentiment][genre] += 1
                    else:
                        genre_counts[sentiment][genre] = 1
        
        # Create a stacked bar chart of genre distribution
        create_genre_distribution_chart(genre_counts)

def display_sentiment_results(sentiment_result, text):
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    
    st.markdown("### Sentiment Analysis Results")
    st.markdown(f"**Text:** {text}")
    
    # Get the max sentiment
    max_sentiment = sentiment_result["max_sentiment"]
    
    # Display the overall sentiment with color coding
    if max_sentiment == "Positive":
        st.markdown(f"**Overall Sentiment:** <span class='sentiment-positive'>{max_sentiment}</span>", unsafe_allow_html=True)
    elif max_sentiment == "Neutral":
        st.markdown(f"**Overall Sentiment:** <span class='sentiment-neutral'>{max_sentiment}</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"**Overall Sentiment:** <span class='sentiment-negative'>{max_sentiment}</span>", unsafe_allow_html=True)
    
    # Create three columns for the sentiment scores
    col1, col2, col3 = st.columns(3)
    
    # Display sentiment probabilities in columns
    with col1:
        pos_value = sentiment_result["probabilities"]["Positive"]
        st.metric(label="Positive", value=f"{pos_value:.2%}")
    
    with col2:
        neu_value = sentiment_result["probabilities"]["Neutral"]
        st.metric(label="Neutral", value=f"{neu_value:.2%}")
    
    with col3:
        neg_value = sentiment_result["probabilities"]["Negative"]
        st.metric(label="Negative", value=f"{neg_value:.2%}")
    
    # Create a bar chart of sentiment probabilities using Plotly
    fig = go.Figure()
    sentiments = ["Positive", "Neutral", "Negative"]
    probabilities = [sentiment_result["probabilities"][s] for s in sentiments]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    
    fig.add_trace(go.Bar(
        x=sentiments,
        y=probabilities,
        text=[f"{p:.1%}" for p in probabilities],
        textposition='auto',
        marker_color=colors
    ))
    
    fig.update_layout(
        title="Sentiment Analysis Scores",
        yaxis=dict(
            title="Probability",
            range=[0, 1]
        ),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def display_media_recommendations(sentiment_result, num_recommendations):
    st.markdown('<p class="section-heading">Personalized Media Recommendations</p>', unsafe_allow_html=True)
    
    st.markdown(f"""
    Based on the sentiment analysis, here are personalized book and movie recommendations 
    that match the emotional tone of the text. The recommendations consider the full sentiment 
    profile with primary emphasis on the dominant sentiment: 
    <span class="sentiment-{sentiment_result['max_sentiment'].lower()}">{sentiment_result['max_sentiment']}</span>.
    """, unsafe_allow_html=True)

    # Get recommendations using weighted approach based on probabilities
    book_recommendations = media_recommender.get_genre_recommendations(
        sentiment_result["probabilities"], "books", num_recommendations
    )
    
    movie_recommendations = media_recommender.get_genre_recommendations(
        sentiment_result["probabilities"], "movies", num_recommendations
    )
    
    # Display book recommendations
    st.markdown("### 📚 Book Recommendations")
    
    # Create a grid layout for books
    cols = st.columns(min(3, len(book_recommendations)))
    for i, book in enumerate(book_recommendations):
        col_idx = i % len(cols)
        with cols[col_idx]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"<p class='media-title'>{book['title']}</p>", unsafe_allow_html=True)
            st.markdown(f"**Author:** {book['author']}")
            st.markdown(f"**Genre:** {book['genre']}")
            st.markdown(f"**Year:** {book['year']}")
            st.markdown(f"**Description:** {book['description']}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Display movie recommendations
    st.markdown("### 🎬 Movie Recommendations")
    
    # Create a grid layout for movies
    cols = st.columns(min(3, len(movie_recommendations)))
    for i, movie in enumerate(movie_recommendations):
        col_idx = i % len(cols)
        with cols[col_idx]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"<p class='media-title'>{movie['title']}</p>", unsafe_allow_html=True)
            st.markdown(f"**Year:** {movie['release_year']}")
            st.markdown(f"**Genre:** {movie['genre']}")
            st.markdown(f"**Rating:** {movie['rating']}/10")
            st.markdown(f"**Description:** {movie['description']}")
            st.markdown('</div>', unsafe_allow_html=True)

def display_genre_mapping(sentiment):
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-heading">Genre Mapping for This Sentiment</p>', unsafe_allow_html=True)
    
    genres = media_recommender.get_genres_for_sentiment(sentiment)
    
    st.markdown(f"""
    The system maps **{sentiment}** sentiment to the following genres. 
    Media in these genres typically evoke or align with {sentiment.lower()} emotional responses:
    """)
    
    # Create a horizontal list of genres
    genre_html = "<div style='display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px;'>"
    for genre in genres:
        genre_html += f"<div style='background-color: #f0f0f0; padding: 6px 12px; border-radius: 16px; font-size: 0.9rem;'>{genre}</div>"
    genre_html += "</div>"
    
    st.markdown(genre_html, unsafe_allow_html=True)
    
    st.markdown("""
    The mapping between sentiments and genres is based on emotional associations commonly found
    in media consumption research. For example, comedy and inspirational content often align with
    positive sentiments, while horror and psychological thrillers tend to align with negative emotions.
    """)

def create_sentiment_pie_chart(results_df):
    # Count the occurrences of each sentiment
    sentiment_counts = results_df["max_sentiment"].value_counts()
    
    # Create a pie chart with Plotly
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Distribution of Sentiments",
        color=sentiment_counts.index,
        color_discrete_map={
            "Positive": "#2ecc71",
            "Neutral": "#3498db",
            "Negative": "#e74c3c"
        }
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(legend_title_text='Sentiment')
    
    st.plotly_chart(fig, use_container_width=True)

def create_sentiment_bar_chart(results_df):
    # Calculate average sentiment scores
    avg_sentiments = {
        "Positive": results_df["positive"].mean(),
        "Neutral": results_df["neutral"].mean(),
        "Negative": results_df["negative"].mean()
    }
    
    # Create bar chart with Plotly
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(avg_sentiments.keys()),
        y=list(avg_sentiments.values()),
        text=[f"{v:.1%}" for v in avg_sentiments.values()],
        textposition='auto',
        marker_color=["#2ecc71", "#3498db", "#e74c3c"]
    ))
    
    fig.update_layout(
        title="Average Sentiment Scores",
        yaxis=dict(
            title="Average Probability",
            range=[0, 1]
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_sentiment_word_clouds(texts, results_df):
    # Create a DataFrame with texts and their sentiments
    text_sentiment_df = pd.DataFrame({
        "text": texts,
        "sentiment": results_df["max_sentiment"]
    })
    
    # Create word clouds for each sentiment
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Positive Text Word Cloud")
        positive_texts = " ".join(text_sentiment_df[text_sentiment_df["sentiment"] == "Positive"]["text"].astype(str))
        if positive_texts.strip():  # Check if there's any text
            create_word_cloud(positive_texts, "#2ecc71")
        else:
            st.info("No positive texts found in the dataset.")
    
    with col2:
        st.markdown("#### Neutral Text Word Cloud")
        neutral_texts = " ".join(text_sentiment_df[text_sentiment_df["sentiment"] == "Neutral"]["text"].astype(str))
        if neutral_texts.strip():  # Check if there's any text
            create_word_cloud(neutral_texts, "#3498db")
        else:
            st.info("No neutral texts found in the dataset.")
    
    with col3:
        st.markdown("#### Negative Text Word Cloud")
        negative_texts = " ".join(text_sentiment_df[text_sentiment_df["sentiment"] == "Negative"]["text"].astype(str))
        if negative_texts.strip():  # Check if there's any text
            create_word_cloud(negative_texts, "#e74c3c")
        else:
            st.info("No negative texts found in the dataset.")

def create_word_cloud(text, hex_color):
    # Define a color function for a single hex color
    def single_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return hex_color

    wordcloud = WordCloud(
        width=400,
        height=300,
        background_color='white',
        max_words=100,
        contour_width=1,
        contour_color='steelblue'
    ).generate(text)

    # Apply the single color
    wordcloud = wordcloud.recolor(color_func=single_color_func)

    # Display the word cloud
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

def create_genre_distribution_chart(genre_counts):
    # Prepare data for visualization
    genres = set()
    for sentiment in genre_counts:
        for genre in genre_counts[sentiment]:
            genres.add(genre)
    
    # Sort genres by frequency across all sentiments
    total_genre_counts = {}
    for genre in genres:
        total_genre_counts[genre] = sum(genre_counts[sentiment].get(genre, 0) for sentiment in genre_counts)
    
    # Get top 15 genres by count
    top_genres = sorted(genres, key=lambda x: total_genre_counts[x], reverse=True)[:15]
    
    # Create data for stacked bar chart
    data = []
    for sentiment in ["Positive", "Neutral", "Negative"]:
        sentiment_data = []
        for genre in top_genres:
            sentiment_data.append(genre_counts[sentiment].get(genre, 0))
        data.append(sentiment_data)
    
    # Create stacked bar chart with Plotly
    fig = go.Figure()
    
    sentiments = ["Positive", "Neutral", "Negative"]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    
    for i, sentiment in enumerate(sentiments):
        fig.add_trace(go.Bar(
            name=sentiment,
            x=top_genres,
            y=data[i],
            marker_color=colors[i]
        ))
    
    fig.update_layout(
        title="Top Genres by Sentiment",
        xaxis_title="Genre",
        yaxis_title="Count",
        barmode='stack',
        height=500,
        legend_title_text='Sentiment'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    This chart shows the distribution of genres across different sentiments. 
    It helps visualize which genres are more commonly associated with positive, neutral, or negative emotions.
    Some genres may appear in multiple sentiment categories but with different frequencies, reflecting their versatility.
    """)

if __name__ == "__main__":
    main()    
