# 🎬📚 Sentiment-Based Movie & Book Recommendation System

## 📌 About the Project

This project is an **AI-powered Movie & Book Recommendation System** that tailors recommendations based on the user's mood and genre preferences.

---

## 🔍 What I Did

- Developed an interactive **Streamlit web app** where users can enter free-form text describing their current mood or interest.
- Integrated a **RoBERTa-based sentiment analysis model** (`cardiffnlp/twitter-roberta-base-sentiment`) to classify user input into **Positive**, **Neutral**, or **Negative** sentiment.
- Used **zero-shot learning** with the `facebook/bart-large-mnli` model to detect the most relevant **genre** based on user input — no custom training required.
- Designed smart logic to generate personalized **movie and book recommendations** using:
  - Sentiment-based filtering (e.g., cheerful content for positive mood, thoughtful reads for negative mood)
  - Genre-matching based on user intent
  - Popularity and rating as tie-breakers or fallbacks
- Integrated **TMDB API** to fetch and display real-time **movie posters**.
- Used two curated datasets:
  - `top10K-TMDB-movies.csv` – Movies with title, genre, overview, popularity, and vote average.
  - `book.csv` – Books with title, author, genre, rating, and synopsis.
- Designed a clean and intuitive UI with:
  - Emoji indicators for sentiment and genre
  - Progress bars to visualize sentiment/genre confidence
  - Expandable sections for movie plots and book synopses

---

## 💡 Key Highlights

- No need to pick from drop-downs — the app understands natural language!
- Users just describe how they feel or what they want to watch/read.
- AI does the rest — understanding the mood, genre, and suggesting the perfect fit.

---

## 🚀 Technologies Used

- Python
- Streamlit
- Hugging Face Transformers
- RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment`)
- BART (`facebook/bart-large-mnli`)
- TMDB API
- Pandas, NumPy, Scikit-learn

---

📝 **Note:** Make sure to include the required dataset files (`top10K-TMDB-movies.csv` and `book.csv`) in the same directory as the app.

