#!/usr/bin/env python3

import pandas as pd
import nltk

from representation_extraction.dataset.dataset import load_dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords


def run(knn: KNeighborsClassifier) -> float:

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

    X_train_texts, y_train = load_dataset("comments_train.txt")
    X_test_texts, y_test = load_dataset("comments_test.txt")

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
    knn.fit(X_train, y_train)

    # --------------------------------------------------
    # Predição
    # --------------------------------------------------
    predictions = knn.predict(X_test)
    probs = knn.predict_proba(X_test)

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
    return float(accuracy_score(y_test, predictions))


if __name__ == "__main__":
    # Cria knn padrão
    knn = KNeighborsClassifier(n_neighbors=7, metric="euclidean")
    run(knn)
