#!/usr/bin/env python3

import pandas as pd
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords

# --------------------------------------------------
# Load NLTK stopwords
# --------------------------------------------------

try:
	stop_words_en = stopwords.words('english')
except LookupError:
	nltk.download('stopwords')
	stop_words_en = stopwords.words('english')
	
# --------------------------------------------------
# Read CSVs
# --------------------------------------------------
	
train_df = pd.read_csv("comments_train.txt")
test_df = pd.read_csv("comments_test.txt")
# --------------------------------------------------
# Dados
# --------------------------------------------------
X_train_texts = train_df["review"].astype(str)
y_train = train_df["label"]

X_test_texts = test_df["review"].astype(str)
y_test = test_df["label"]

# --------------------------------------------------
# TF-IDF
# --------------------------------------------------
print ("Extraindo representacao...")
vectorizer = TfidfVectorizer(
	stop_words=stop_words_en,
	max_features=350,
	ngram_range=(1,2),
	lowercase=True,
	min_df=2, max_df=.9
)

X_train = vectorizer.fit_transform(X_train_texts)
X_test  = vectorizer.transform(X_test_texts)

feature_names = vectorizer.get_feature_names_out()


# --------------------------------------------------
# Modelo
# --------------------------------------------------
print ("Classificando com kNN...")

model = KNeighborsClassifier(n_neighbors=7, metric="euclidean")
model.fit(X_train, y_train)

# --------------------------------------------------
# Predição
# --------------------------------------------------
predictions = model.predict(X_test)
#probs = model.predict_proba(X_test)

# --------------------------------------------------
# Avaliação
# --------------------------------------------------
from sklearn.metrics import classification_report
print("\nEvaluation:")
print(classification_report(y_test, predictions))
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predictions)
print (cm)

