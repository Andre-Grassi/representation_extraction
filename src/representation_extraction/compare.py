from pathlib import Path

import numpy as np
import os
from enum import Enum
import argparse

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import sparse

from representation_extraction.dataset.dataset import load_dataset
from representation_extraction.extractors import tfidf, bow, bert

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = PROJECT_ROOT / "features"

# Extratores disponiveis e se suas features sao esparsas
EXTRACTORS = {
    "tfidf": {"module": tfidf, "sparse": True},
    "bow": {"module": bow, "sparse": True},
    "bert": {"module": bert, "sparse": False},
}


# Enum para indicar se eh teste ou validacao
class ResultType(Enum):
    TEST = "test"
    VALIDATION = "validation"


def export_results_to_csv(
    representation: str,
    result_type: ResultType,
    accuracy: float,
):
    """
    Exporta os resultados do experimento para um arquivo CSV na pasta 'results/{result_type}/{representation}.csv'.

    Args:
        representation (str): Nome do metodo de extracao de caracteristicas.
        result_type (ResultType): Tipo de resultado (teste ou validacao).
        accuracy (float): Acuracia obtida.
    """
    import pandas as pd

    results_dir = PROJECT_ROOT / "results" / result_type.value
    os.makedirs(results_dir, exist_ok=True)

    results_df = pd.DataFrame(
        {
            "Representation": [representation],
            "Accuracy": [accuracy],
        }
    )

    csv_path = results_dir / f"{representation}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"  Resultado salvo em: {csv_path}")


# --------------------------------------------------
# Funcoes de salvar/carregar features
# --------------------------------------------------


def _save_features(extractor_name: str, name: str, X):
    """Salva features (sparse ou dense) na pasta features/<extractor_name>/."""
    out_dir = FEATURES_DIR / extractor_name
    os.makedirs(out_dir, exist_ok=True)

    is_sparse = EXTRACTORS[extractor_name]["sparse"]

    if is_sparse:
        sparse.save_npz(out_dir / f"{name}.npz", X)
    else:
        np.save(out_dir / f"{name}.npy", X)


def _save_labels(extractor_name: str, name: str, y):
    """Salva labels como .npy na pasta features/<extractor_name>/."""
    out_dir = FEATURES_DIR / extractor_name
    os.makedirs(out_dir, exist_ok=True)
    np.save(out_dir / f"{name}.npy", np.array(y))


def _load_features(extractor_name: str, name: str):
    """Carrega features (sparse ou dense) da pasta features/<extractor_name>/."""
    feat_dir = FEATURES_DIR / extractor_name
    is_sparse = EXTRACTORS[extractor_name]["sparse"]

    if is_sparse:
        return sparse.load_npz(feat_dir / f"{name}.npz")
    else:
        return np.load(feat_dir / f"{name}.npy")


def _load_labels(extractor_name: str, name: str):
    """Carrega labels da pasta features/<extractor_name>/."""
    feat_dir = FEATURES_DIR / extractor_name
    return np.load(feat_dir / f"{name}.npy", allow_pickle=True)


# --------------------------------------------------
# Etapas do pipeline
# --------------------------------------------------


def do_extract(extractor_name: str):
    """
    Etapa de extracao: carrega dataset, extrai features, faz split 80/20 estratificado,
    e salva 6 arquivos em features/<extractor_name>/.
    """
    print(f"\n{'='*60}")
    print(f"  EXTRACAO - {extractor_name.upper()}")
    print(f"{'='*60}")

    ext_info = EXTRACTORS[extractor_name]
    ext_module = ext_info["module"]

    # Carrega dataset
    print("Carregando dataset...")
    X_train_texts, y_train = load_dataset("comments_train.txt")
    X_test_texts, y_test = load_dataset("comments_test.txt")

    # Para BERT, precisa converter para lista
    if extractor_name == "bert":
        X_train_texts = X_train_texts.tolist()
        X_test_texts = X_test_texts.tolist()

    # Extrai features
    print(f"Extraindo features com {extractor_name}...")
    X_train_full, X_test = ext_module.extract(X_train_texts, X_test_texts)

    # Split estratificado 80/20 para validacao
    print("Fazendo split 80/20 estratificado...")
    y_train_arr = np.array(y_train)

    if ext_info["sparse"]:
        X_train, X_val, y_train_split, y_val = train_test_split(
            X_train_full, y_train_arr,
            test_size=0.20, random_state=42, stratify=y_train_arr,
        )
    else:
        X_train, X_val, y_train_split, y_val = train_test_split(
            X_train_full, y_train_arr,
            test_size=0.20, random_state=42, stratify=y_train_arr,
        )

    # Salva os 6 arquivos
    print(f"Salvando features em features/{extractor_name}/...")
    _save_features(extractor_name, "X_train", X_train)
    _save_labels(extractor_name, "y_train", y_train_split)
    _save_features(extractor_name, "X_val", X_val)
    _save_labels(extractor_name, "y_val", y_val)
    _save_features(extractor_name, "X_test", X_test)
    _save_labels(extractor_name, "y_test", np.array(y_test))

    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_val shape:   {X_val.shape}")
    print(f"  X_test shape:  {X_test.shape}")
    print(f"  Extracao concluida!\n")


def do_validate(extractor_name: str):
    """
    Etapa de validacao: carrega X_train, X_val da pasta features/<extractor_name>/,
    treina KNN no train (80%), avalia no val (20%).
    """
    print(f"\n{'='*60}")
    print(f"  VALIDACAO - {extractor_name.upper()}")
    print(f"{'='*60}")

    X_train = _load_features(extractor_name, "X_train")
    y_train = _load_labels(extractor_name, "y_train")
    X_val = _load_features(extractor_name, "X_val")
    y_val = _load_labels(extractor_name, "y_val")

    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_val shape:   {X_val.shape}")

    knn = KNeighborsClassifier(n_neighbors=7, metric="euclidean")
    print("Treinando KNN...")
    knn.fit(X_train, y_train)

    print("Predizendo no conjunto de validacao...")
    predictions = knn.predict(X_val)
    accuracy = float(accuracy_score(y_val, predictions))

    print(f"  Acuracia de VALIDACAO: {accuracy:.4f}")

    export_results_to_csv(
        representation=extractor_name,
        result_type=ResultType.VALIDATION,
        accuracy=accuracy,
    )


def do_test(extractor_name: str):
    """
    Etapa de teste: carrega X_train, X_test da pasta features/<extractor_name>/,
    treina KNN no train (80%), avalia no test.
    """
    print(f"\n{'='*60}")
    print(f"  TESTE - {extractor_name.upper()}")
    print(f"{'='*60}")

    X_train = _load_features(extractor_name, "X_train")
    y_train = _load_labels(extractor_name, "y_train")
    X_test = _load_features(extractor_name, "X_test")
    y_test = _load_labels(extractor_name, "y_test")

    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape:  {X_test.shape}")

    knn = KNeighborsClassifier(n_neighbors=7, metric="euclidean")
    print("Treinando KNN...")
    knn.fit(X_train, y_train)

    print("Predizendo no conjunto de teste...")
    predictions = knn.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))

    print(f"  Acuracia de TESTE: {accuracy:.4f}")

    export_results_to_csv(
        representation=extractor_name,
        result_type=ResultType.TEST,
        accuracy=accuracy,
    )


def do_all(extractor_name: str):
    """Roda extract -> validate -> test em sequencia."""
    do_extract(extractor_name)
    do_validate(extractor_name)
    do_test(extractor_name)


# --------------------------------------------------
# Dispatcher: resolve --extractor all
# --------------------------------------------------


def _run_stage(stage_fn, extractor_name: str):
    """Executa a funcao de etapa para um extrator ou todos."""
    if extractor_name == "all":
        for name in EXTRACTORS:
            stage_fn(name)
    else:
        stage_fn(extractor_name)


# --------------------------------------------------
# Argparse
# --------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline de extracao, validacao e teste de representacoes textuais.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Exemplos de uso:
  python compare.py extract  --extractor tfidf
  python compare.py validate --extractor bert
  python compare.py test     --extractor bow
  python compare.py all      --extractor all
""",
    )

    subparsers = parser.add_subparsers(dest="stage", required=True)

    extractor_choices = list(EXTRACTORS.keys()) + ["all"]

    for name, help_text in [
        ("extract", "Extrai features e salva em disco"),
        ("validate", "Valida usando features pre-extraidas (split 80/20)"),
        ("test", "Testa usando features pre-extraidas"),
        ("all", "Roda extract -> validate -> test em sequencia"),
    ]:
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument(
            "--extractor",
            required=True,
            choices=extractor_choices,
            help="Extrator a usar (ou 'all' para todos)",
        )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    stage_map = {
        "extract": do_extract,
        "validate": do_validate,
        "test": do_test,
        "all": do_all,
    }

    stage_fn = stage_map[args.stage]
    _run_stage(stage_fn, args.extractor)


if __name__ == "__main__":
    main()
