#!/usr/bin/env python3

import pandas as pd
import nltk

from representation_extraction.dataset.dataset import load_dataset

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords

from transformers import AutoModel, AutoTokenizer
import torch


def extract_features(model, tokenizer, texts: list[str]):
    model_inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        model_outputs = model(**model_inputs)

    embeddings_cls = model_outputs.last_hidden_state[:, 0, :]
    X_bert = embeddings_cls.cpu().numpy()
    return X_bert


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

    X_train_texts, y_train = load_dataset("comments_train_few.txt")
    X_train_texts_bert = X_train_texts.tolist()
    y_train_bert = y_train.tolist()
    X_test_texts, y_test = load_dataset("comments_test_few.txt")
    X_test_texts_bert = X_test_texts.tolist()
    y_test_bert = y_test.tolist()

    # --------------------------------------------------
    # Carrega o BERT
    # --------------------------------------------------
    print("Carregando BERT...")
    model = AutoModel.from_pretrained(
        "bert-base-uncased", dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # --------------------------------------------------
    # BERT
    # --------------------------------------------------
    print("Extraindo representacao...")
    X_train_bert = extract_features(model, tokenizer, X_train_texts_bert)
    X_test_bert = extract_features(model, tokenizer, X_test_texts_bert)
    knn.fit(X_train_bert, y_train)

    # --------------------------------------------------
    # Predição
    # --------------------------------------------------
    predictions = knn.predict(X_test_bert)
    probs = knn.predict_proba(X_test_bert)

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
    accuracy = run(knn)
    print(f"Acurácia: {accuracy}")
