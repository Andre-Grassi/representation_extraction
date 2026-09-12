from enum import Enum
from pathlib import Path
from sklearn.preprocessing import MaxAbsScaler, Normalizer

from representation_extraction.extractors import tfidf, bow, bert

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = PROJECT_ROOT / "features"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"

# Extratores disponiveis e se suas features sao esparsas
EXTRACTORS = {
    "tfidf": {"module": tfidf, "sparse": True},
    "bow": {"module": bow, "sparse": True},
    "bert": {"module": bert, "sparse": False},
}

# Hiperparametros do extrator: quais args pertencem a cada extrator
EXTRACTOR_PARAMS = {
    "tfidf": {"max_features", "ngram_range", "min_df", "max_df", "normalization"},
    "bow": {"max_features", "ngram_range", "min_df", "max_df", "normalization"},
    "bert": {"max_length", "model_name", "normalization"},
}

# Defaults dos hiperparametros dos extratores
EXTRACTOR_DEFAULTS = {
    "tfidf": {
        "max_features": 350,
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.9,
        "normalization": "none",
    },
    "bow": {
        "max_features": 350,
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.9,
        "normalization": "none",
    },
    "bert": {
        "max_length": 16,
        "model_name": "bert-base-uncased",
        "normalization": "none",
    },
}

# Defaults do KNN
KNN_DEFAULTS = {"n_neighbors": 7, "metric": "euclidean", "weights": "uniform"}

# --------------------------------------------------
# Grids para o Grid Search (param_grid para GridSearchCV com Pipeline)
# --------------------------------------------------
GRIDSEARCH_PARAM_GRIDS = {
    "bow": {
        "extractor__max_features": [250, 500, 1000],
        "extractor__ngram_range": [(1, 1), (1, 2), (1, 3)],
        "extractor__min_df": [1, 2, 3],
        "extractor__max_df": [0.8, 0.9, 1.0],
        "scaler": [MaxAbsScaler(), Normalizer()],
        "knn__n_neighbors": [3, 7, 11],
        "knn__metric": ["euclidean", "cosine"],
        "knn__weights": ["uniform", "distance"],
    },
    "tfidf": {
        "extractor__max_features": [250, 500, 1000],
        "extractor__ngram_range": [(1, 1), (1, 2), (1, 3)],
        "extractor__min_df": [1, 2, 3],
        "extractor__max_df": [0.8, 0.9, 1.0],
        "scaler": [MaxAbsScaler(), Normalizer()],
        "knn__n_neighbors": [3, 7, 11],
        "knn__metric": ["euclidean", "cosine"],
        "knn__weights": ["uniform", "distance"],
    },
    "bert": {
        "extractor__max_length": [16, 32, 64],
        "extractor__model_name": [
            "bert-base-uncased",
            "bert-large-uncased",
            "distilbert-base-uncased",
        ],
        "scaler": [MaxAbsScaler(), Normalizer()],
        "knn__n_neighbors": [3, 7, 11],
        "knn__metric": ["euclidean", "cosine"],
        "knn__weights": ["uniform", "distance"],
    },
}

# Enum para indicar se eh teste ou validacao
class ResultType(Enum):
    TEST = "test"
    VALIDATION = "validation"
