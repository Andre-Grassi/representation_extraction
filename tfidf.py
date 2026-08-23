#!/usr/bin/env python3

import pandas as pd
import nltk

from dataset import load_dataset
from compare import export_results_to_csv, ResultType

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords

# --------------------------------------------------
# Load NLTK stopwords
# --------------------------------------------------

try:
    stop_words_en = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")
    stop_words_en = stopwords.words("english")

# --------------------------------------------------
# Carrega datasets
# --------------------------------------------------

X_train_texts, y_train = load_dataset("dataset/comments_train.txt")
X_test_texts, y_test = load_dataset("dataset/comments_test.txt")

# --------------------------------------------------
# TF-IDF
# --------------------------------------------------
print("Extraindo representacao...")
vectorizer = TfidfVectorizer(
    stop_words=stop_words_en,
    max_features=350,
    ngram_range=(1, 2),
    lowercase=True,
    min_df=2,
    max_df=0.9,
)

X_train = vectorizer.fit_transform(X_train_texts)
X_test = vectorizer.transform(X_test_texts)

feature_names = vectorizer.get_feature_names_out()


# --------------------------------------------------
# Modelo
# --------------------------------------------------
print("Classificando com kNN...")

model = KNeighborsClassifier(n_neighbors=7, metric="euclidean")
model.fit(X_train, y_train)

# --------------------------------------------------
# Predição
# --------------------------------------------------
predictions = model.predict(X_test)
probs = model.predict_proba(X_test)

# --------------------------------------------------
# Avaliação
# --------------------------------------------------
from sklearn.metrics import accuracy_score, classification_report

# print("\nEvaluation:")
# print(classification_report(y_test, predictions))
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predictions)
# print(probs)

# Pega acurácia do TESTE
test_accuracy = accuracy_score(y_test, predictions)
print("Acurácia do TESTE: {:.4f}".format(test_accuracy))
export_results_to_csv(
    representation="tfidf",
    result_type=ResultType.TEST,
    accuracy_validation=0.0,
    accuracy_test=float(test_accuracy),
    execution_time=0.0,
)
