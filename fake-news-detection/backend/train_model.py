import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, confusion_matrix

# Define dataset path
DATASET_PATH = "DataSet.csv"
MODEL_DIR = "model"

# Ensure model directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Load dataset
df = pd.read_csv(DATASET_PATH)

# Check dataset balance before handling imbalance
print("Before Balancing:", df['label'].value_counts())

# Convert labels (REAL = 1, FAKE = 0)
df['label'] = df['label'].map({'REAL': 1, 'FAKE': 0})

# Handle missing values
df.dropna(subset=['text', 'label'], inplace=True)

# Separate REAL and FAKE news
real_news = df[df['label'] == 1]
fake_news = df[df['label'] == 0]

# Balance dataset
if abs(len(real_news) - len(fake_news)) > 10:  # Only balance if the difference is significant
    if len(fake_news) < len(real_news):
        real_news = resample(real_news, replace=False, n_samples=len(fake_news), random_state=42)
    else:
        fake_news = resample(fake_news, replace=False, n_samples=len(real_news), random_state=42)

df = pd.concat([real_news, fake_news]).sample(frac=1, random_state=42)

# Check dataset balance after handling imbalance
print("After Balancing:", df['label'].value_counts())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Model (Passive Aggressive Classifier)
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_tfidf, y_train)

# Evaluate Model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:\n", conf_matrix)

# Save Model & Vectorizer
with open(os.path.join(MODEL_DIR, "fake_news_model.pkl"), "wb") as model_file:
    pickle.dump(model, model_file)

with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("✅ Model training complete. Saved in 'model/' directory.")
