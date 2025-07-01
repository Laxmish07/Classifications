import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Dummy tweets related to Queen Elizabeth II's death
dummy_tweets = [
    "The world mourns the loss of Queen Elizabeth II. Her reign will be remembered for generations to come. #RIPQueenElizabeth",
    "Sad news today as we learn about the passing of Queen Elizabeth II. Her legacy will forever be etched in history. #QueenElizabeth",
    "Rest in peace, Queen Elizabeth II. Your grace, dignity, and devotion to duty will never be forgotten. #RIP",
    "The loss of Queen Elizabeth II is deeply felt around the world. She was a symbol of strength and resilience. #QueenElizabeth",
    "Heartbroken to hear about the passing of Queen Elizabeth II. She was a true leader and inspiration to us all. #RIPQueen",
    "Celebrating the incredible life of Queen Elizabeth II. Her legacy will continue to inspire us. #QueenElizabeth",
    "Queen Elizabeth II's death is a tragic loss for the world. She will be deeply missed. #RIPQueenElizabeth",
    "Honoring the memory of Queen Elizabeth II. Her grace and leadership will never be forgotten. #QueenElizabeth",
    "Today marks the end of an era as we bid farewell to Queen Elizabeth II. Her legacy will live on in the hearts of millions. #RIPQueen",
    "Deeply saddened by the news of Queen Elizabeth II's passing. Her grace and dedication to her people were unmatched. #QueenElizabeth",
    "A sad day for the world as we mourn the loss of Queen Elizabeth II. She was a true symbol of strength and resilience. #RIP",
    "We have lost a remarkable leader in Queen Elizabeth II. Her reign will be remembered as a time of progress and unity. #QueenElizabeth",
    "Honoring the memory of Queen Elizabeth II, who served her country with unwavering dedication. Rest in peace. #RIPQueenElizabeth",
    "Today, we say goodbye to Queen Elizabeth II, a beloved monarch and an inspiration to us all. May her soul rest in peace. #QueenElizabeth",
    "The passing of Queen Elizabeth II leaves a void in our hearts. She will forever be remembered for her grace and compassion. #RIPQueen",
    "In memory of Queen Elizabeth II, whose reign was marked by compassion, wisdom, and service to her nation. Rest in peace. #QueenElizabeth"
]

# Sentiment labels (0 for negative, 1 for positive)
sentiment_labels = [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1]

# Initialize lemmatizer and begin the text preprocessing steps
lemmatizer = WordNetLemmatizer()

# Tokenize and lemmatize tweets
tokenized_tweets = [word_tokenize(tweet.lower()) for tweet in dummy_tweets]
lemmatized_tweets = [[lemmatizer.lemmatize(word) for word in tweet] for tweet in tokenized_tweets]

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_tweets = [[word for word in tweet if word not in stop_words] for tweet in lemmatized_tweets]

# Convert filtered tweets back to string
preprocessed_tweets = [' '.join(tweet) for tweet in filtered_tweets]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(preprocessed_tweets, sentiment_labels, test_size=0.2, random_state=42)

# Create TF-IDF vectorizer to convert text form to numeric form
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

# Predict sentiment on test data
y_pred = classifier.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Naive Bayes Accuracy:", accuracy)

# Print sentiment predictions for test data
print("\nNaive Bayes Sentiment Predictions:")
for i, tweet in enumerate(X_test):
    sentiment = "Positive" if y_pred[i] == 1 else "Negative"
    print("Tweet:", tweet)
    print("Predicted Sentiment:", sentiment)
    print("Actual Sentiment:", "Positive" if y_test[i] == 1 else "Negative")

# Sentiment analysis using TextBlob
print("\nTextBlob Sentiment Predictions:")
for tweet, label in zip(dummy_tweets, sentiment_labels):
    analysis = TextBlob(tweet)
    polarity = analysis.sentiment.polarity
    predicted_sentiment = "Positive" if polarity > 0 else "Negative"
    print("Tweet:", tweet)
    print("Predicted Sentiment:", predicted_sentiment)
    print("Actual Sentiment:", "Positive" if label == 1 else "Negative")

# Sentiment analysis using VADER
print("\nVADER Sentiment Predictions:")
sid = SentimentIntensityAnalyzer()
for tweet, label in zip(dummy_tweets, sentiment_labels):
    scores = sid.polarity_scores(tweet)
    compound_score = scores['compound']
    predicted_sentiment = "Positive" if compound_score >= 0 else "Negative"
    print("Tweet:", tweet)
    print("Predicted Sentiment:", predicted_sentiment)
    print("Actual Sentiment:", "Positive" if label == 1 else "Negative")
