#!/usr/bin/env python3
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix

from representation_extraction.dataset.dataset import load_dataset
from .base import BaseExtractor

def _get_stopwords():
    try:
        return stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")
        return stopwords.words("english")

class TfidfExtractor(BaseExtractor):
    def __init__(
        self,
        max_features: int = 350,
        ngram_range: tuple = (1, 2),
        min_df: int = 2,
        max_df: float = 0.9,
        normalization: str = "none"
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.normalization = normalization
        self.vectorizer_ = None

    def fit(self, X, y=None):
        stop_words_en = _get_stopwords()
        self.vectorizer_ = TfidfVectorizer(
            stop_words=stop_words_en,
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            lowercase=True,
            min_df=self.min_df,
            max_df=self.max_df,
        )
        self.vectorizer_.fit(X)
        return self

    def transform(self, X):
        return self.vectorizer_.transform(X)


def extract(
    X_train_texts,
    X_test_texts,
    max_features: int = 350,
    ngram_range: tuple = (1, 2),
    min_df: int = 2,
    max_df: float = 0.9,
) -> tuple[csr_matrix, csr_matrix]:
    extractor = TfidfExtractor(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
    )
    return extractor.extract(X_train_texts, X_test_texts)


def run(knn: KNeighborsClassifier) -> float:
    X_train_texts, y_train = load_dataset("comments_train.txt")
    X_test_texts, y_test = load_dataset("comments_test.txt")

    print("Extraindo representacao...")
    extractor = TfidfExtractor()
    X_train, X_test = extractor.extract(X_train_texts, X_test_texts)

    print("Classificando com kNN...")
    knn.fit(X_train, y_train)

    predictions = knn.predict(X_test)
    
    from sklearn.metrics import accuracy_score
    return float(accuracy_score(y_test, predictions))

if __name__ == "__main__":
    knn = KNeighborsClassifier(n_neighbors=7, metric="euclidean")
    run(knn)
