import pandas as pd
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = PROJECT_ROOT / "dataset"


def load_dataset(file_name: str) -> tuple[pd.Series, pd.Series]:
    """
    Carrega um dataset a partir de um arquivo de texto no formato csv.
    """
    file_path = Path(file_name)
    if not file_path.is_absolute():
        file_path = DATASET_DIR / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"O arquivo {file_path} nao foi encontrado.")

    df = pd.read_csv(file_path)

    texts = df["review"].astype(str)
    labels = df["label"]

    return (texts, labels)


def normalize_features(X_train, X_val, X_test, method: str):
    """
    Aplica a normalizacao nas features extraidas. O fit e realizado
    exclusivamente no conjunto de treino, e o transform e aplicado
    no treino, validacao e teste.
    """
    if method == "none":
        return X_train, X_val, X_test
        
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
        
    from sklearn.preprocessing import MaxAbsScaler, Normalizer

    if method == "maxabs":
        scaler = MaxAbsScaler()
    elif method == "normalizer":
        scaler = Normalizer()
    else:
        raise ValueError(f"Metodo de normalizacao desconhecido: {method}")

    # O fit DEVE ocorrer somente no treino para evitar vazamento de dados
    scaler.fit(X_train)

    X_train_norm = scaler.transform(X_train)
    X_val_norm = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test)

    return X_train_norm, X_val_norm, X_test_norm
