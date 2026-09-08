#!/usr/bin/env python3

import pandas as pd
import numpy as np
import nltk

from representation_extraction.dataset.dataset import load_dataset

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords

from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm


def extract_features(model, tokenizer, texts: list[str], batch_size: int = 32):

    # Extrai embeddings de batch_size em batch_size para evitar estouro de memória
    all_embeddings = []
    all_texts = len(texts)

    for i in tqdm(range(0, all_texts, batch_size), desc="Extraindo embeddings BERT"):
        batch_texts = texts[i : i + batch_size]

        model_inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            model_outputs = model(**model_inputs)

        embeddings_cls = model_outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(embeddings_cls)

        # Libera memória cache do PyTorch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    X_bert = np.vstack(all_embeddings)
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
    X_train = extract_features(model, tokenizer, X_train_texts_bert)
    X_test = extract_features(model, tokenizer, X_test_texts_bert)
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
    accuracy = run(knn)
    print(f"Acurácia: {accuracy}")
