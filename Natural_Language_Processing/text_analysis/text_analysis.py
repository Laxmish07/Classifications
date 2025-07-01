import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv("E:\CSU Work\Thesis and RA\dataset_4\emotion_recog.csv", encoding_errors='ignore')
# Filter out emotions with less than 50 tags
emotion_counts = df['sentiment'].value_counts()
valid_emotions = emotion_counts[emotion_counts >= 50].index
df = df[df['sentiment'].isin(valid_emotions)]

# Preprocessing function
def preprocess_text(text):
    # Remove Twitter handles
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenization
    tokens = word_tokenize(text)
    # Lowercasing and lemmatization
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.lower()]
    # Stemming
    #tokens = [stemmer.stem(word) for word in tokens]
    # Join tokens back into string
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
# Initialize stemmer , lemmatization gives better accuracy than stemming
#stemmer = PorterStemmer()
#stop_words = set(stopwords.words('english'))

# Apply preprocessing to 'content' column
df['preprocessed_content'] = df['content'].apply(preprocess_text)
df.to_csv("preprocessed_text_analysis.csv",index=True)
# Define feature and target variables
X = df['preprocessed_content']
y = df['sentiment']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize and train Multiclass Logistic Regression model
logistic_regression_model = LogisticRegression(max_iter=3000)
logistic_regression_model.fit(X_train_tfidf, y_train)

# Predict sentiment on test data
y_pred_logistic_regression = logistic_regression_model.predict(X_test_tfidf)

# Calculate accuracy for Multiclass Logistic Regression
accuracy_logistic_regression = accuracy_score(y_test, y_pred_logistic_regression)
print("Multiclass Logistic Regression Accuracy:", accuracy_logistic_regression)

# Print classification report for Multiclass Logistic Regression
print("Classification Report for Multiclass Logistic Regression:")
print(classification_report(y_test, y_pred_logistic_regression,zero_division=0))

# Plot confusion matrix for Multiclass Logistic Regression
def plot_confusion_matrix(y_true, y_pred,model):
    labels = model.classes_
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix :' + model.__class__.__name__)
    plt.savefig("cm_"+model.__class__.__name__+".png")

plot_confusion_matrix(y_test, y_pred_logistic_regression,logistic_regression_model)

# Initialize and train Gaussian Naïve Bayes model
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train_tfidf.toarray(), y_train)

# Predict sentiment on test data
y_pred_naive_bayes = naive_bayes_model.predict(X_test_tfidf.toarray())

# Calculate accuracy for Gaussian Naïve Bayes
accuracy_naive_bayes = accuracy_score(y_test, y_pred_naive_bayes)
print("Gaussian Naïve Bayes Accuracy:", accuracy_naive_bayes)

# Print classification report for Gaussian Naïve Bayes
print("Classification Report for Gaussian Naïve Bayes:")
print(classification_report(y_test, y_pred_naive_bayes,zero_division=0))

# Plot confusion matrix for Gaussian Naïve Bayes
plot_confusion_matrix(y_test, y_pred_naive_bayes,naive_bayes_model)
