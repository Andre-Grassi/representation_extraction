from pathlib import Path

import json
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

# Hiperparametros do extrator: quais args pertencem a cada extrator
EXTRACTOR_PARAMS = {
    "tfidf": {"max_features", "ngram_range", "min_df", "max_df"},
    "bow": {"max_features", "ngram_range", "min_df", "max_df"},
    "bert": {"max_length", "model_name"},
}

# Defaults dos hiperparametros dos extratores
EXTRACTOR_DEFAULTS = {
    "tfidf": {"max_features": 350, "ngram_range": (1, 2), "min_df": 2, "max_df": 0.9},
    "bow": {"max_features": 350, "ngram_range": (1, 2), "min_df": 2, "max_df": 0.9},
    "bert": {"max_length": 16, "model_name": "bert-base-uncased"},
}

# Defaults do KNN
KNN_DEFAULTS = {"n_neighbors": 7, "metric": "euclidean", "weights": "uniform"}


# Enum para indicar se eh teste ou validacao
class ResultType(Enum):
    TEST = "test"
    VALIDATION = "validation"


# --------------------------------------------------
# Funcoes utilitarias de hiperparametros
# --------------------------------------------------


def _build_folder_name(extractor_name: str, ext_params: dict) -> str:
    """Gera o nome da subpasta a partir dos hiperparametros do extrator."""
    if extractor_name in ("tfidf", "bow"):
        mf = ext_params["max_features"]
        ng_min, ng_max = ext_params["ngram_range"]
        md = ext_params["min_df"]
        mxd = ext_params["max_df"]
        return f"mf{mf}_ng{ng_min}-{ng_max}_md{md}_mxd{mxd}"
    elif extractor_name == "bert":
        ml = ext_params["max_length"]
        mn = ext_params["model_name"]
        return f"ml{ml}_{mn}"
    else:
        raise ValueError(f"Extrator desconhecido: {extractor_name}")


def _get_extractor_params(extractor_name: str, args) -> dict:
    """Extrai os hiperparametros do extrator a partir dos args do argparse."""
    defaults = EXTRACTOR_DEFAULTS[extractor_name]
    params = {}
    for key in defaults:
        val = getattr(args, key, None)
        if val is not None:
            params[key] = val
        else:
            params[key] = defaults[key]
    # Converte ngram_range de lista para tupla se necessario
    if "ngram_range" in params and isinstance(params["ngram_range"], list):
        params["ngram_range"] = tuple(params["ngram_range"])
    return params


def _get_knn_params(args) -> dict:
    """Extrai os hiperparametros do KNN a partir dos args do argparse."""
    params = {}
    for key in KNN_DEFAULTS:
        val = getattr(args, key, None)
        if val is not None:
            params[key] = val
        else:
            params[key] = KNN_DEFAULTS[key]
    return params


def _validate_extractor_args(extractor_name: str, args):
    """Verifica se o usuario nao passou hiperparametros invalidos para o extrator."""
    if extractor_name == "all":
        return  # Nao valida quando e 'all' (usa defaults)

    valid_params = EXTRACTOR_PARAMS[extractor_name]
    all_extractor_params = set()
    for params in EXTRACTOR_PARAMS.values():
        all_extractor_params.update(params)

    for param in all_extractor_params:
        if param not in valid_params:
            val = getattr(args, param, None)
            default = None
            # Checa se o usuario setou explicitamente um param que nao pertence ao extrator
            if val is not None:
                raise ValueError(
                    f"Hiperparametro '--{param.replace('_', '-')}' nao se aplica ao extrator '{extractor_name}'. "
                    f"Hiperparametros validos para '{extractor_name}': {sorted(valid_params)}"
                )


def _save_params_json(features_dir: Path, extractor_name: str, ext_params: dict):
    """Salva params.json na pasta de features."""
    params_to_save = {"extractor": extractor_name}
    for k, v in ext_params.items():
        # Converte tuplas para listas para serializacao JSON
        if isinstance(v, tuple):
            params_to_save[k] = list(v)
        else:
            params_to_save[k] = v
    with open(features_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(params_to_save, f, indent=2)


# --------------------------------------------------
# Funcoes de exportacao de resultados
# --------------------------------------------------


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


def _save_features(extractor_name: str, feat_dir: Path, name: str, X):
    """Salva features (sparse ou dense) na pasta de features."""
    os.makedirs(feat_dir, exist_ok=True)
    is_sparse = EXTRACTORS[extractor_name]["sparse"]

    if is_sparse:
        sparse.save_npz(feat_dir / f"{name}.npz", X)
    else:
        np.save(feat_dir / f"{name}.npy", X)


def _save_labels(feat_dir: Path, name: str, y):
    """Salva labels como .npy na pasta de features."""
    os.makedirs(feat_dir, exist_ok=True)
    np.save(feat_dir / f"{name}.npy", np.array(y))


def _load_features(extractor_name: str, feat_dir: Path, name: str):
    """Carrega features (sparse ou dense) da pasta de features."""
    is_sparse = EXTRACTORS[extractor_name]["sparse"]

    if is_sparse:
        return sparse.load_npz(feat_dir / f"{name}.npz")
    else:
        return np.load(feat_dir / f"{name}.npy")


def _load_labels(feat_dir: Path, name: str):
    """Carrega labels da pasta de features."""
    return np.load(feat_dir / f"{name}.npy", allow_pickle=True)


# --------------------------------------------------
# Etapas do pipeline
# --------------------------------------------------


def do_extract(extractor_name: str, args):
    """
    Etapa de extracao: carrega dataset, extrai features, faz split 80/20 estratificado,
    e salva 6 arquivos em features/<extractor_name>/<folder_name>/.
    """
    ext_params = _get_extractor_params(extractor_name, args)
    folder_name = _build_folder_name(extractor_name, ext_params)
    feat_dir = FEATURES_DIR / extractor_name / folder_name

    print(f"\n{'='*60}")
    print(f"  EXTRACAO - {extractor_name.upper()}")
    print(f"  Pasta: features/{extractor_name}/{folder_name}/")
    print(f"  Params: {ext_params}")
    print(f"{'='*60}")

    ext_module = EXTRACTORS[extractor_name]["module"]

    # Carrega dataset
    print("Carregando dataset...")
    X_train_texts, y_train = load_dataset("comments_train.txt")
    X_test_texts, y_test = load_dataset("comments_test.txt")

    # Para BERT, precisa converter para lista
    if extractor_name == "bert":
        X_train_texts = X_train_texts.tolist()
        X_test_texts = X_test_texts.tolist()

    # Extrai features com hiperparametros
    print(f"Extraindo features com {extractor_name}...")
    if extractor_name in ("tfidf", "bow"):
        X_train_full, X_test = ext_module.extract(
            X_train_texts, X_test_texts,
            max_features=ext_params["max_features"],
            ngram_range=ext_params["ngram_range"],
            min_df=ext_params["min_df"],
            max_df=ext_params["max_df"],
        )
    elif extractor_name == "bert":
        X_train_full, X_test = ext_module.extract(
            X_train_texts, X_test_texts,
            max_length=ext_params["max_length"],
            model_name=ext_params["model_name"],
        )

    # Split estratificado 80/20 para validacao
    print("Fazendo split 80/20 estratificado...")
    y_train_arr = np.array(y_train)

    X_train, X_val, y_train_split, y_val = train_test_split(
        X_train_full, y_train_arr,
        test_size=0.20, random_state=42, stratify=y_train_arr,
    )

    # Salva os 6 arquivos
    print(f"Salvando features em features/{extractor_name}/{folder_name}/...")
    _save_features(extractor_name, feat_dir, "X_train", X_train)
    _save_labels(feat_dir, "y_train", y_train_split)
    _save_features(extractor_name, feat_dir, "X_val", X_val)
    _save_labels(feat_dir, "y_val", y_val)
    _save_features(extractor_name, feat_dir, "X_test", X_test)
    _save_labels(feat_dir, "y_test", np.array(y_test))

    # Salva params.json
    _save_params_json(feat_dir, extractor_name, ext_params)

    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_val shape:   {X_val.shape}")
    print(f"  X_test shape:  {X_test.shape}")
    print(f"  Extracao concluida!\n")


def do_validate(extractor_name: str, args):
    """
    Etapa de validacao: carrega X_train, X_val da pasta features/<extractor_name>/<folder>/,
    treina KNN no train (80%), avalia no val (20%).
    """
    ext_params = _get_extractor_params(extractor_name, args)
    knn_params = _get_knn_params(args)
    folder_name = _build_folder_name(extractor_name, ext_params)
    feat_dir = FEATURES_DIR / extractor_name / folder_name

    if not feat_dir.exists():
        raise FileNotFoundError(
            f"Pasta de features nao encontrada: {feat_dir}\n"
            f"Execute primeiro: python compare.py extract --extractor {extractor_name} "
            f"(com os mesmos hiperparametros)"
        )

    print(f"\n{'='*60}")
    print(f"  VALIDACAO - {extractor_name.upper()}")
    print(f"  Pasta: features/{extractor_name}/{folder_name}/")
    print(f"  KNN params: {knn_params}")
    print(f"{'='*60}")

    X_train = _load_features(extractor_name, feat_dir, "X_train")
    y_train = _load_labels(feat_dir, "y_train")
    X_val = _load_features(extractor_name, feat_dir, "X_val")
    y_val = _load_labels(feat_dir, "y_val")

    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_val shape:   {X_val.shape}")

    knn = KNeighborsClassifier(**knn_params)
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


def do_test(extractor_name: str, args):
    """
    Etapa de teste: carrega X_train, X_test da pasta features/<extractor_name>/<folder>/,
    treina KNN no train (80%), avalia no test.
    """
    ext_params = _get_extractor_params(extractor_name, args)
    knn_params = _get_knn_params(args)
    folder_name = _build_folder_name(extractor_name, ext_params)
    feat_dir = FEATURES_DIR / extractor_name / folder_name

    if not feat_dir.exists():
        raise FileNotFoundError(
            f"Pasta de features nao encontrada: {feat_dir}\n"
            f"Execute primeiro: python compare.py extract --extractor {extractor_name} "
            f"(com os mesmos hiperparametros)"
        )

    print(f"\n{'='*60}")
    print(f"  TESTE - {extractor_name.upper()}")
    print(f"  Pasta: features/{extractor_name}/{folder_name}/")
    print(f"  KNN params: {knn_params}")
    print(f"{'='*60}")

    X_train = _load_features(extractor_name, feat_dir, "X_train")
    y_train = _load_labels(feat_dir, "y_train")
    X_test = _load_features(extractor_name, feat_dir, "X_test")
    y_test = _load_labels(feat_dir, "y_test")

    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape:  {X_test.shape}")

    knn = KNeighborsClassifier(**knn_params)
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


def do_all(extractor_name: str, args):
    """Roda extract -> validate -> test em sequencia."""
    do_extract(extractor_name, args)
    do_validate(extractor_name, args)
    do_test(extractor_name, args)


# --------------------------------------------------
# Dispatcher: resolve --extractor all
# --------------------------------------------------


def _run_stage(stage_fn, extractor_name: str, args):
    """Executa a funcao de etapa para um extrator ou todos."""
    if extractor_name == "all":
        for name in EXTRACTORS:
            stage_fn(name, args)
    else:
        stage_fn(extractor_name, args)


# --------------------------------------------------
# Argparse
# --------------------------------------------------


def _add_extractor_args(parser):
    """Adiciona argumentos de hiperparametros dos extratores ao parser."""
    # BoW / TF-IDF
    parser.add_argument(
        "--max-features", type=int, default=None,
        help="Numero maximo de features (bow/tfidf, default: 350)",
    )
    parser.add_argument(
        "--ngram-range", type=int, nargs=2, default=None, metavar=("MIN", "MAX"),
        help="Range de n-grams (bow/tfidf, default: 1 2)",
    )
    parser.add_argument(
        "--min-df", type=int, default=None,
        help="Frequencia minima de documento (bow/tfidf, default: 2)",
    )
    parser.add_argument(
        "--max-df", type=float, default=None,
        help="Frequencia maxima de documento (bow/tfidf, default: 0.9)",
    )
    # BERT
    parser.add_argument(
        "--max-length", type=int, default=None,
        help="Comprimento maximo da sequencia de tokens (bert, default: 16)",
    )
    parser.add_argument(
        "--model-name", type=str, default=None,
        help="Nome do modelo HuggingFace (bert, default: bert-base-uncased)",
    )


def _add_knn_args(parser):
    """Adiciona argumentos de hiperparametros do KNN ao parser."""
    parser.add_argument(
        "--n-neighbors", type=int, default=None,
        help="Numero de vizinhos do KNN (default: 7)",
    )
    parser.add_argument(
        "--metric", type=str, default=None,
        help="Metrica de distancia do KNN (default: euclidean)",
    )
    parser.add_argument(
        "--weights", type=str, default=None, choices=["uniform", "distance"],
        help="Peso dos vizinhos do KNN (default: uniform)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline de extracao, validacao e teste de representacoes textuais.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Exemplos de uso:
  python compare.py extract  --extractor tfidf
  python compare.py extract  --extractor tfidf --max-features 500 --ngram-range 1 3
  python compare.py extract  --extractor bert --max-length 64 --model-name distilbert-base-uncased
  python compare.py validate --extractor tfidf --n-neighbors 5 --metric cosine
  python compare.py test     --extractor bow
  python compare.py all      --extractor all
""",
    )

    subparsers = parser.add_subparsers(dest="stage", required=True)

    extractor_choices = list(EXTRACTORS.keys()) + ["all"]

    # extract: hiper. do extrator
    sub_extract = subparsers.add_parser("extract", help="Extrai features e salva em disco")
    sub_extract.add_argument(
        "--extractor", required=True, choices=extractor_choices,
        help="Extrator a usar (ou 'all' para todos)",
    )
    _add_extractor_args(sub_extract)

    # validate: hiper. do extrator (para localizar pasta) + hiper. do KNN
    sub_validate = subparsers.add_parser("validate", help="Valida usando features pre-extraidas (split 80/20)")
    sub_validate.add_argument(
        "--extractor", required=True, choices=extractor_choices,
        help="Extrator a usar (ou 'all' para todos)",
    )
    _add_extractor_args(sub_validate)
    _add_knn_args(sub_validate)

    # test: hiper. do extrator (para localizar pasta) + hiper. do KNN
    sub_test = subparsers.add_parser("test", help="Testa usando features pre-extraidas")
    sub_test.add_argument(
        "--extractor", required=True, choices=extractor_choices,
        help="Extrator a usar (ou 'all' para todos)",
    )
    _add_extractor_args(sub_test)
    _add_knn_args(sub_test)

    # all: hiper. do extrator + hiper. do KNN
    sub_all = subparsers.add_parser("all", help="Roda extract -> validate -> test em sequencia")
    sub_all.add_argument(
        "--extractor", required=True, choices=extractor_choices,
        help="Extrator a usar (ou 'all' para todos)",
    )
    _add_extractor_args(sub_all)
    _add_knn_args(sub_all)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Valida hiperparametros do extrator
    if args.extractor != "all":
        _validate_extractor_args(args.extractor, args)

    stage_map = {
        "extract": do_extract,
        "validate": do_validate,
        "test": do_test,
        "all": do_all,
    }

    stage_fn = stage_map[args.stage]
    _run_stage(stage_fn, args.extractor, args)


if __name__ == "__main__":
    main()
