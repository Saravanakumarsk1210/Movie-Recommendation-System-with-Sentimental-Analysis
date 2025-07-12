# 🎥📚 **Movie and Book Recommendations with Genre Mapping and Twitter Sentiment Analysis**

This project provides an intelligent, mood-aware content discovery tool that analyzes Twitter-like text input and recommends movies and books based on detected sentiment. The system utilizes a RoBERTa-based sentiment analyzer and a genre mapping engine to connect emotional tone with personalized media suggestions.

---

## 🚀 Features

- **RoBERTa Twitter Sentiment Analyzer**
  - Detects Positive, Neutral, and Negative sentiments
  - Outputs sentiment probabilities and dominant sentiment

- **Genre Mapping System**
  - Maps sentiment → genres
  - E.g., Positive → Comedy, Romance; Negative → Horror, Psychological

- **Media Recommendation Engine**
  - Recommends mock book/movie data based on sentiment
  - Includes title, author/director, genre, year, and description

- **Interactive Streamlit Interface**
  - Analyze a single text input and get instant media suggestions
  - Upload CSV for batch analysis and visualize sentiment insights
  - Explore genre mapping logic

---

## 🖥️ Application Preview

![image](https://github.com/user-attachments/assets/9bdaf8d2-646a-4998-983c-49978b54b802)
![image](https://github.com/user-attachments/assets/01bcb8d6-3929-49c6-b168-48fe24fa3abd)
![image](https://github.com/user-attachments/assets/3ca9d096-9779-420c-a104-df9fceeb8dc8)
![image](https://github.com/user-attachments/assets/4ef3a810-a55c-419f-9d40-a937caefcd40)
![image](https://github.com/user-attachments/assets/ad12d532-3f10-45e8-ab9f-37cc15e91d16)
![image](https://github.com/user-attachments/assets/34590bbf-563d-4573-83d1-89083d62c23c)
![image](https://github.com/user-attachments/assets/9414597a-06a7-4886-8691-181f19bb3235)

<!--
---

## 🖥️ Deployed Streamlit Link

https://movie-and-book-recommendations-with-tweets.streamlit.app/
-->
---

## 📂 Project Structure

```
📁 project-root/
│
├── app.py              # Streamlit app
├── sentiment.py        # Sentiment analysis and recommendation logic
├── requirements.txt    # Python dependencies
└── README.md           # Project overview

```
---

## 📦 Installation

1. **Clone the repository**:

```bash
git clone https://github.com/mvharsh/Movie-and-Book-Recommendations.git
cd Movie-and-Book-Recommendations

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Run the application**:

```bash
streamlit run app.py
```

---

## 🔍 Key Components

### 1. Sentiment Analysis
- Uses `cardiffnlp/twitter-roberta-base-sentiment`
- Preprocesses input tweets for user mentions and links
- Outputs probabilities for Positive, Neutral, and Negative

### 2. Genre Mapping
- Sentiments are associated with emotional genres:
    - **Positive**: Comedy, Romance, Family, Animation...
    - **Neutral**: Sci-Fi, Documentary, History...
    - **Negative**: Horror, Psychological, Thriller...

### 3. Recommendations
- **Mock APIs** simulate data from TMDb and OpenLibrary
- Recommendations based on:
    - Dominant sentiment
    - Weighted sentiment profile

---

## 📊 Streamlit App Features

| Feature                    | Description |
|---------------------------|-------------|
| **Single Text Analysis**  | Analyze one tweet or post and get media recs |
| **Batch CSV Analysis**    | Upload CSV of tweets, analyze all at once, download results |
| **Genre Mapping Explorer**| Understand how emotions map to genres |

---

## 📈 Visualizations

- Sentiment Probability Bar Charts
- Pie Charts for batch sentiment distribution
- Word Cloud per sentiment
- Stacked Genre Frequency Charts

---

## 📁 Example CSV for Batch Analysis

```csv
id,text
1,"I just had an amazing coffee and I feel great!"
2,"The weather is so gloomy, it's making me sad."
3,"Watched a really thought-provoking documentary today."
```

---

## ▶️ Youtube Link

https://www.youtube.com/watch?v=zh6AuxX8JVQ

