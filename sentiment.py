# sentiment_analysis.py
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import pandas as pd
import numpy as np
import random
import time

class SentimentAnalyzer:
    def __init__(self):
        # Define the pre-trained model and labels
        self.MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
        self.LABELS = ['Negative', 'Neutral', 'Positive']
        
        # Load the model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
    
    def preprocess_tweet(self, tweet):
        """
        Preprocesses a tweet by replacing user mentions and URLs.
        
        Parameters:
        tweet (str): Original tweet text.
        
        Returns:
        str: Preprocessed tweet text.
        """
        tweet_words = []
        for word in tweet.split(' '):
            if word.startswith('@') and len(word) > 1:
                word = '@user'
            elif word.startswith('http'):
                word = "http"
            tweet_words.append(word)
        return " ".join(tweet_words)
    
    def analyze_sentiment(self, tweet):
        """
        Analyzes sentiment of a tweet using a pre-trained RoBERTa model.
        
        Parameters:
        tweet (str): Tweet text.
        
        Returns:
        dict: Sentiment probabilities and max sentiment.
        """
        tweet_proc = self.preprocess_tweet(tweet)
        encoded_tweet = self.tokenizer(tweet_proc, return_tensors='pt')
        output = self.model(**encoded_tweet)
        scores = softmax(output[0][0].detach().numpy())
        sentiment_probabilities = {self.LABELS[i]: float(scores[i]) for i in range(len(self.LABELS))}
        
        # Determine the dominant sentiment
        max_sentiment = max(sentiment_probabilities, key=sentiment_probabilities.get)
        
        return {
            "probabilities": sentiment_probabilities,
            "max_sentiment": max_sentiment
        }

class MediaRecommender:
    def __init__(self):
        # Genre mapping based on sentiment
        self.genre_sentiment_mapping = {
            "Positive": [
                "Comedy", "Animation", "Family", "Adventure", "Musical", 
                "Romance", "Fantasy", "Inspirational", "Biography", "Sport"
            ],
            "Neutral": [
                "Documentary", "History", "Science Fiction", "Action", "Western",
                "Mystery", "Drama", "Superhero", "Animation", "Thriller"
            ],
            "Negative": [
                "Horror", "Thriller", "Crime", "War", "Drama", 
                "Mystery", "Film-Noir", "Psychological", "Disaster", "Dystopian"
            ]
        }
        
        # Fetch initial dataset - will be cached
        self._fetch_movie_data()
        self._fetch_book_data()
        
    def _fetch_movie_data(self):
        """Fetch movie data from TMDb API"""
        # We'll use TMDb API (free tier) for movies
        self.movies = {}
        
        # Try to load from local cache first - implement caching to avoid API rate limits
        try:
            # Mock data to avoid hitting API limits
            self.movies = {
                "Positive": self._get_mock_positive_movies(),
                "Neutral": self._get_mock_neutral_movies(),
                "Negative": self._get_mock_negative_movies()
            }
        except Exception as e:
            print(f"Error fetching movie data: {e}")
            # Fallback to empty dataset
            self.movies = {"Positive": [], "Neutral": [], "Negative": []}
    
    def _fetch_book_data(self):
        """Fetch book data from OpenLibrary API"""
        # We'll use OpenLibrary for books data
        self.books = {}
        
        # Try to load from local cache first
        try:
            # Mock data to avoid hitting API limits
            self.books = {
                "Positive": self._get_mock_positive_books(),
                "Neutral": self._get_mock_neutral_books(),
                "Negative": self._get_mock_negative_books()
            }
        except Exception as e:
            print(f"Error fetching book data: {e}")
            # Fallback to empty dataset
            self.books = {"Positive": [], "Neutral": [], "Negative": []}
    
    def _get_mock_positive_movies(self):
        """Mock positive sentiment movies"""
        return [
            {"title": "The Pursuit of Happyness", "release_year": 2006, "genre": "Drama, Biography", "rating": 8.0, 
             "description": "A struggling salesman takes custody of his son as he's poised to begin a life-changing professional career."},
            {"title": "La La Land", "release_year": 2016, "genre": "Musical, Romance", "rating": 8.0, 
             "description": "While navigating their careers in Los Angeles, a pianist and an actress fall in love while attempting to reconcile their aspirations for the future."},
            {"title": "Soul", "release_year": 2020, "genre": "Animation, Adventure", "rating": 8.1, 
             "description": "A musician who has lost his passion for music is transported out of his body and must find his way back with the help of an infant soul learning about herself."},
            {"title": "Forrest Gump", "release_year": 1994, "genre": "Drama, Romance", "rating": 8.8, 
             "description": "The presidencies of Kennedy and Johnson, the Vietnam War, the Watergate scandal and other historical events unfold from the perspective of an Alabama man with an IQ of 75."},
            {"title": "Toy Story", "release_year": 1995, "genre": "Animation, Adventure", "rating": 8.3, 
             "description": "A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him as top toy in a boy's room."},
            {"title": "Inside Out", "release_year": 2015, "genre": "Animation, Adventure", "rating": 8.1, 
             "description": "After young Riley is uprooted from her Midwest life and moved to San Francisco, her emotions - Joy, Fear, Anger, Disgust and Sadness - conflict on how best to navigate a new city, house, and school."},
            {"title": "CODA", "release_year": 2021, "genre": "Drama, Music", "rating": 8.0, 
             "description": "As a CODA (Child of Deaf Adults), Ruby is the only hearing person in her deaf family. When the family's fishing business is threatened, Ruby finds herself torn between pursuing her love of music and her fear of abandoning her parents."},
            {"title": "Love Actually", "release_year": 2003, "genre": "Comedy, Drama", "rating": 7.6, 
             "description": "Follows the lives of eight very different couples in dealing with their love lives in various loosely interrelated tales all set during a frantic month before Christmas in London, England."},
            {"title": "The Intouchables", "release_year": 2011, "genre": "Biography, Comedy", "rating": 8.5, 
             "description": "After he becomes a quadriplegic from a paragliding accident, an aristocrat hires a young man from the projects to be his caregiver."},
            {"title": "Up", "release_year": 2009, "genre": "Animation, Adventure", "rating": 8.2, 
             "description": "78-year-old Carl Fredricksen travels to Paradise Falls in his house equipped with balloons, inadvertently taking a young stowaway."}
        ]
    
    def _get_mock_neutral_movies(self):
        """Mock neutral sentiment movies"""
        return [
            {"title": "Inception", "release_year": 2010, "genre": "Action, Adventure", "rating": 8.8, 
             "description": "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O."},
            {"title": "The Social Network", "release_year": 2010, "genre": "Biography, Drama", "rating": 7.7, 
             "description": "As Harvard student Mark Zuckerberg creates the social networking site that would become known as Facebook, he is sued by the twins who claimed he stole their idea, and by the co-founder who was later squeezed out of the business."},
            {"title": "The Martian", "release_year": 2015, "genre": "Adventure, Drama", "rating": 8.0, 
             "description": "An astronaut becomes stranded on Mars after his team assume him dead, and must rely on his ingenuity to find a way to signal to Earth that he is alive."},
            {"title": "The Matrix", "release_year": 1999, "genre": "Action, Sci-Fi", "rating": 8.7, 
             "description": "When a beautiful stranger leads computer hacker Neo to a forbidding underworld, he discovers the shocking truth--the life he knows is the elaborate deception of an evil cyber-intelligence."},
            {"title": "Avatar", "release_year": 2009, "genre": "Action, Adventure", "rating": 7.8, 
             "description": "A paraplegic Marine dispatched to the moon Pandora on a unique mission becomes torn between following his orders and protecting the world he feels is his home."},
            {"title": "Dune", "release_year": 2021, "genre": "Action, Adventure", "rating": 8.0, 
             "description": "Feature adaptation of Frank Herbert's science fiction novel about the son of a noble family entrusted with the protection of the most valuable asset and most vital element in the galaxy."},
            {"title": "Casino Royale", "release_year": 2006, "genre": "Action, Adventure", "rating": 8.0, 
             "description": "After earning 00 status and a licence to kill, Secret Agent James Bond sets out on his first mission as 007. Bond must defeat a private banker funding terrorists in a high-stakes game of poker at Casino Royale, Montenegro."},
            {"title": "Interstellar", "release_year": 2014, "genre": "Adventure, Drama", "rating": 8.6, 
             "description": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival."},
            {"title": "The Prestige", "release_year": 2006, "genre": "Drama, Mystery", "rating": 8.5, 
             "description": "After a tragic accident, two stage magicians engage in a battle to create the ultimate illusion while sacrificing everything they have to outwit each other."},
            {"title": "Memento", "release_year": 2000, "genre": "Mystery, Thriller", "rating": 8.4, 
             "description": "A man with short-term memory loss attempts to track down his wife's murderer."}
        ]
    
    def _get_mock_negative_movies(self):
        """Mock negative sentiment movies"""
        return [
            {"title": "Joker", "release_year": 2019, "genre": "Crime, Drama", "rating": 8.4, 
             "description": "In Gotham City, mentally troubled comedian Arthur Fleck is disregarded and mistreated by society. He then embarks on a downward spiral of revolution and bloody crime. This path brings him face-to-face with his alter-ego: the Joker."},
            {"title": "The Lighthouse", "release_year": 2019, "genre": "Drama, Fantasy", "rating": 7.5, 
             "description": "Two lighthouse keepers try to maintain their sanity while living on a remote and mysterious New England island in the 1890s."},
            {"title": "Requiem for a Dream", "release_year": 2000, "genre": "Drama", "rating": 8.3, 
             "description": "The drug-induced utopias of four Coney Island people are shattered when their addictions run deep."},
            {"title": "Black Swan", "release_year": 2010, "genre": "Drama, Thriller", "rating": 8.0, 
             "description": "A committed dancer struggles to maintain her sanity after winning the lead role in a production of Tchaikovsky's 'Swan Lake'."},
            {"title": "No Country for Old Men", "release_year": 2007, "genre": "Crime, Drama", "rating": 8.1, 
             "description": "Violence and mayhem ensue after a hunter stumbles upon a drug deal gone wrong and more than two million dollars in cash near the Rio Grande."},
            {"title": "Hereditary", "release_year": 2018, "genre": "Drama, Horror", "rating": 7.3, 
             "description": "A grieving family is haunted by tragic and disturbing occurrences after the death of their secretive grandmother."},
            {"title": "Uncut Gems", "release_year": 2019, "genre": "Crime, Drama", "rating": 7.4, 
             "description": "With his debts mounting and angry collectors closing in, a fast-talking New York City jeweler risks everything in hope of staying afloat and alive."},
            {"title": "The Revenant", "release_year": 2015, "genre": "Action, Adventure", "rating": 8.0, 
             "description": "A frontiersman on a fur trading expedition in the 1820s fights for survival after being mauled by a bear and left for dead by members of his own hunting team."},
            {"title": "Se7en", "release_year": 1995, "genre": "Crime, Drama", "rating": 8.6, 
             "description": "Two detectives, a rookie and a veteran, hunt a serial killer who uses the seven deadly sins as his motives."},
            {"title": "The Silence of the Lambs", "release_year": 1991, "genre": "Crime, Drama", "rating": 8.6, 
             "description": "A young F.B.I. cadet must receive the help of an incarcerated and manipulative cannibal killer to help catch another serial killer, a madman who skins his victims."}
        ]
    
    def _get_mock_positive_books(self):
        """Mock positive sentiment books"""
        return [
            {"title": "The Alchemist", "author": "Paulo Coelho", "genre": "Fiction, Adventure", "year": 1988, 
             "description": "A story about following your dreams and listening to your heart."},
            {"title": "Atomic Habits", "author": "James Clear", "genre": "Self-Help, Productivity", "year": 2018, 
             "description": "Tiny Changes, Remarkable Results: An Easy & Proven Way to Build Good Habits & Break Bad Ones."},
            {"title": "The Power of Positive Thinking", "author": "Norman Vincent Peale", "genre": "Self-Help", "year": 1952, 
             "description": "A practical guide to mastering the problems of everyday living."},
            {"title": "Man's Search for Meaning", "author": "Viktor E. Frankl", "genre": "Psychology, Memoir", "year": 1946, 
             "description": "Psychiatrist Viktor Frankl's memoir has riveted generations with its descriptions of life in Nazi death camps and its lessons for spiritual survival."},
            {"title": "The Book of Joy", "author": "Dalai Lama and Desmond Tutu", "genre": "Spirituality", "year": 2016, 
             "description": "Lasting Happiness in a Changing World."},
            {"title": "A Man Called Ove", "author": "Fredrik Backman", "genre": "Fiction, Humor", "year": 2012, 
             "description": "A grumpy yet loveable man finds his solitary world turned on its head when a boisterous young family moves in next door."},
            {"title": "Where the Crawdads Sing", "author": "Delia Owens", "genre": "Fiction, Coming of Age", "year": 2018, 
             "description": "For years, rumors of the 'Marsh Girl' have haunted Barkley Cove, a quiet town on the North Carolina coast."},
            {"title": "The Happiness Project", "author": "Gretchen Rubin", "genre": "Self-Help, Memoir", "year": 2009, 
             "description": "Or, Why I Spent a Year Trying to Sing in the Morning, Clean My Closets, Fight Right, Read Aristotle, and Generally Have More Fun."},
            {"title": "Little Women", "author": "Louisa May Alcott", "genre": "Fiction, Classic", "year": 1868, 
             "description": "The story of the lives of the four March sisters—Meg, Jo, Beth, and Amy—detailing their passage from childhood to womanhood."},
            {"title": "Anne of Green Gables", "author": "L.M. Montgomery", "genre": "Fiction, Children's", "year": 1908, 
             "description": "The adventures of an 11-year-old orphan girl who lives on Prince Edward Island."}
        ]
    
    def _get_mock_neutral_books(self):
        """Mock neutral sentiment books"""
        return [
            {"title": "Sapiens: A Brief History of Humankind", "author": "Yuval Noah Harari", "genre": "History, Science", "year": 2011, 
             "description": "A brief history of humankind from the Stone Age up to the twenty-first century."},
            {"title": "Thinking, Fast and Slow", "author": "Daniel Kahneman", "genre": "Psychology, Economics", "year": 2011, 
             "description": "How the human mind works, and how we make decisions."},
            {"title": "Educated", "author": "Tara Westover", "genre": "Memoir, Biography", "year": 2018, 
             "description": "A memoir about a young girl who, kept out of school, leaves her survivalist family and goes on to earn a PhD from Cambridge University."},
            {"title": "The Silent Patient", "author": "Alex Michaelides", "genre": "Mystery, Thriller", "year": 2019, 
             "description": "A psychological thriller about a woman's act of violence against her husband―and of the therapist obsessed with uncovering her motive."},
            {"title": "1984", "author": "George Orwell", "genre": "Fiction, Dystopian", "year": 1949, 
             "description": "A dystopian social science fiction novel set in a world of perpetual war, omnipresent government surveillance, and public manipulation."},
            {"title": "The Great Gatsby", "author": "F. Scott Fitzgerald", "genre": "Fiction, Classic", "year": 1925, 
             "description": "A portrait of the Jazz Age in all of its decadence and excess."},
            {"title": "To Kill a Mockingbird", "author": "Harper Lee", "genre": "Fiction, Classic", "year": 1960, 
             "description": "A novel about the childhood of Scout Finch in a Southern town and her father's battle for justice."},
            {"title": "The Catcher in the Rye", "author": "J.D. Salinger", "genre": "Fiction, Coming of Age", "year": 1951, 
             "description": "The story of a teenaged boy dealing with alienation."},
            {"title": "Pride and Prejudice", "author": "Jane Austen", "genre": "Fiction, Romance", "year": 1813, 
             "description": "A romantic novel of manners that follows the character development of Elizabeth Bennet."},
            {"title": "The Hobbit", "author": "J.R.R. Tolkien", "genre": "Fiction, Fantasy", "year": 1937, 
             "description": "A fantasy novel about the adventures of hobbit Bilbo Baggins."}
        ]
    
    def _get_mock_negative_books(self):
        """Mock negative sentiment books"""
        return [
            {"title": "The Road", "author": "Cormac McCarthy", "genre": "Fiction, Post-Apocalyptic", "year": 2006, 
             "description": "A journey of a father and his son walking alone through burned America, heading through the ravaged landscape to the coast."},
            {"title": "Crime and Punishment", "author": "Fyodor Dostoevsky", "genre": "Fiction, Psychological", "year": 1866, 
             "description": "The story of the mental anguish and moral dilemmas of Rodion Raskolnikov, an impoverished ex-student in Saint Petersburg who formulates a plan to kill an unscrupulous pawnbroker."},
            {"title": "The Bell Jar", "author": "Sylvia Plath", "genre": "Fiction, Autobiographical", "year": 1963, 
             "description": "Chronicles a young woman's descent into mental illness."},
            {"title": "No Longer Human", "author": "Osamu Dazai", "genre": "Fiction, Psychological", "year": 1948, 
             "description": "The poignant and fascinating story of a young man who is caught between the breakup of the traditions of a northern Japanese aristocratic family and the impact of Western ideas."},
            {"title": "The Metamorphosis", "author": "Franz Kafka", "genre": "Fiction, Absurdist", "year": 1915, 
             "description": "A novella about a man who wakes up one morning to find himself transformed into a huge insect."},
            {"title": "Brave New World", "author": "Aldous Huxley", "genre": "Fiction, Dystopian", "year": 1932, 
             "description": "A dystopian novel set in a futuristic World State of genetically modified citizens and an intelligence-based social hierarchy."},
            {"title": "The Stranger", "author": "Albert Camus", "genre": "Fiction, Philosophical", "year": 1942, 
             "description": "Through the story of an ordinary man unwittingly drawn into a senseless murder on an Algerian beach, Camus explored what he termed 'the nakedness of man faced with the absurd.'"},
            {"title": "Lord of the Flies", "author": "William Golding", "genre": "Fiction, Allegorical", "year": 1954, 
             "description": "A group of British boys stuck on an uninhabited island who try to govern themselves with disastrous results."},
            {"title": "A Clockwork Orange", "author": "Anthony Burgess", "genre": "Fiction, Dystopian", "year": 1962, 
             "description": "A frightening fable about good and evil, and the meaning of human freedom."},
            {"title": "Pet Sematary", "author": "Stephen King", "genre": "Fiction, Horror", "year": 1983, 
             "description": "When Dr. Louis Creed takes a new job and moves his family to the idyllic rural town of Ludlow, Maine, this new beginning seems too good to be true."}
        ]

    def get_recommendations(self, sentiment, media_type, num_recommendations=3):
        """
        Get recommendations based on sentiment and media type.
        
        Parameters:
        sentiment (str): Sentiment category (Positive, Neutral, Negative)
        media_type (str): Type of media ('books' or 'movies')
        num_recommendations (int): Number of recommendations to return
        
        Returns:
        list: List of recommended items
        """
        if media_type == 'books':
            all_recommendations = self.books.get(sentiment, [])
        elif media_type == 'movies':
            all_recommendations = self.movies.get(sentiment, [])
        else:
            return []
        
        # If we have more recommendations than requested, randomly select a subset
        if len(all_recommendations) > num_recommendations:
            return random.sample(all_recommendations, num_recommendations)
        return all_recommendations
    
    def get_genre_recommendations(self, sentiment_probs, media_type, num_recommendations=5):
        """
        Get weighted recommendations based on sentiment probabilities.
        This provides more nuanced recommendations based on the full sentiment profile.
        
        Parameters:
        sentiment_probs (dict): Sentiment probabilities (e.g., {"Positive": 0.7, "Neutral": 0.2, "Negative": 0.1})
        media_type (str): Type of media ('books' or 'movies')
        num_recommendations (int): Number of recommendations to return
        
        Returns:
        list: List of recommended items
        """
        # Get items from each sentiment category based on probability weights
        if media_type == 'books':
            all_items = self.books
        elif media_type == 'movies':
            all_items = self.movies
        else:
            return []
        
        # Calculate how many items to take from each sentiment category
        total_items = []
        for sentiment, prob in sentiment_probs.items():
            # Calculate number of items to take (at least 1 if probability > 0.1)
            items_to_take = max(1, int(round(num_recommendations * prob))) if prob > 0.1 else 0
            available_items = all_items.get(sentiment, [])
            
            if available_items and items_to_take > 0:
                # Take random items from this sentiment category
                selected = random.sample(available_items, min(items_to_take, len(available_items)))
                total_items.extend(selected)
        
        # If we still don't have enough items, add more from the highest probability sentiment
        if len(total_items) < num_recommendations:
            max_sentiment = max(sentiment_probs, key=sentiment_probs.get)
            available_items = all_items.get(max_sentiment, [])
            
            # Filter out items already selected
            available_items = [item for item in available_items if item not in total_items]
            
            if available_items:
                remaining_needed = num_recommendations - len(total_items)
                additional_items = random.sample(available_items, min(remaining_needed, len(available_items)))
                total_items.extend(additional_items)
        
        # If we have too many items, take a random subset
        if len(total_items) > num_recommendations:
            total_items = random.sample(total_items, num_recommendations)
            
        return total_items
    
    def get_genres_for_sentiment(self, sentiment):
        """Get the genres associated with a particular sentiment."""
        return self.genre_sentiment_mapping.get(sentiment, [])