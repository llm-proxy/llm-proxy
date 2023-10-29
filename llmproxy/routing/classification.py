import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Load the CSV data into a DataFrame
data = pd.read_csv('text_data.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['prompts'], data['category'], test_size=0.2, random_state=42)

# Create TF-IDF vectors
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Train a Multinomial Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# Function to classify user input
def classify_input(user_input):
    # Preprocess the user input
    user_input_tfidf = tfidf_vectorizer.transform([user_input])

    # Predict the category
    category = nb_classifier.predict(user_input_tfidf)[0]
    return category

if __name__ == "__main__":
    # Evaluate the model
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    y_pred = nb_classifier.predict(X_test_tfidf)
    print(classification_report(y_test, y_pred))
