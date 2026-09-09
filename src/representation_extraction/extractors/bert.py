#!/usr/bin/env python3

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm


def _extract_embeddings(model, tokenizer, texts: list[str], batch_size: int = 32, max_length: int = 16):
    """
    Extrai CLS embeddings em lotes para evitar estouro de memoria.
    """
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Extraindo embeddings BERT"):
        batch_texts = texts[i : i + batch_size]

        model_inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            model_outputs = model(**model_inputs)

        embeddings_cls = model_outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(embeddings_cls.cpu().numpy())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.vstack(all_embeddings)


class BertTransformer:
    """
    Transformer sklearn-compativel que encapsula a extracao de features BERT.
    Implementa fit/transform para funcionar dentro de um Pipeline + GridSearchCV.
    Tambem e usado internamente pela funcao extract().
    """

    def __init__(self, max_length=16, model_name="bert-base-uncased"):
        self.max_length = max_length
        self.model_name = model_name

    def fit(self, X, y=None):
        self.model_ = AutoModel.from_pretrained(
            self.model_name, dtype="auto", device_map="auto"
        )
        self.tokenizer_ = AutoTokenizer.from_pretrained(self.model_name)
        return self

    def transform(self, X):
        texts = list(X) if not isinstance(X, list) else X
        return _extract_embeddings(
            self.model_, self.tokenizer_, texts, max_length=self.max_length
        )

    def get_params(self, deep=True):
        return {"max_length": self.max_length, "model_name": self.model_name}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


def extract(
    X_train_texts: list[str],
    X_test_texts: list[str],
    max_length: int = 16,
    model_name: str = "bert-base-uncased",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrai features BERT (CLS embeddings) dos textos de treino e teste.
    Usa o BertTransformer internamente.
    """
    transformer = BertTransformer(max_length=max_length, model_name=model_name)
    transformer.fit(X_train_texts)

    X_train = transformer.transform(X_train_texts)
    X_test = transformer.transform(X_test_texts)

    return X_train, X_test


if __name__ == "__main__":
    from representation_extraction.dataset.dataset import load_dataset

    X_train_texts, y_train = load_dataset("comments_train.txt")
    X_test_texts, y_test = load_dataset("comments_test.txt")

    knn = KNeighborsClassifier(n_neighbors=7, metric="euclidean")

    X_train, X_test = extract(X_train_texts.tolist(), X_test_texts.tolist())
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)

    from sklearn.metrics import accuracy_score
    print(f"Acuracia: {accuracy_score(y_test, predictions):.4f}")
