from pathlib import Path

import json
import numpy as np
import os
from enum import Enum
import argparse
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import sparse

from representation_extraction.dataset.dataset import load_dataset
from representation_extraction.extractors import tfidf, bow, bert

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = PROJECT_ROOT / "features"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"

from representation_extraction.statistics.statistics import get_statistics

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

from sklearn.preprocessing import MaxAbsScaler, Normalizer

# --------------------------------------------------
# Grids para o Grid Search (param_grid para GridSearchCV com Pipeline)
# Prefixos: extractor__ para hiperparametros do extrator, knn__ para KNN.
# O passo 'scaler' testa diferentes métodos de normalização.
# --------------------------------------------------

GRIDSEARCH_PARAM_GRIDS = {
    "bow": {
        "extractor__max_features": [100, 250, 500, 1000],
        "extractor__ngram_range": [(1, 1), (1, 2), (1, 3)],
        "extractor__min_df": [1, 2, 3],
        "extractor__max_df": [0.8, 0.9, 1.0],
        "scaler": [MaxAbsScaler(), Normalizer()],
        "knn__n_neighbors": [3, 5, 7, 9, 11],
        "knn__metric": ["euclidean", "cosine", "manhattan"],
        "knn__weights": ["uniform", "distance"],
    },
    "tfidf": {
        "extractor__max_features": [100, 250, 500, 1000],
        "extractor__ngram_range": [(1, 1), (1, 2), (1, 3)],
        "extractor__min_df": [1, 2, 3],
        "extractor__max_df": [0.8, 0.9, 1.0],
        "scaler": [MaxAbsScaler(), Normalizer()],
        "knn__n_neighbors": [3, 5, 7, 9, 11],
        "knn__metric": ["euclidean", "cosine", "manhattan"],
        "knn__weights": ["uniform", "distance"],
    },
    "bert": {
        "scaler": [MaxAbsScaler(), Normalizer()],
        "knn__n_neighbors": [3, 5, 7, 9, 11],
        "knn__metric": ["euclidean", "cosine", "manhattan"],
        "knn__weights": ["uniform", "distance"],
    },
}


# Enum para indicar se eh teste ou validacao
class ResultType(Enum):
    TEST = "test"
    VALIDATION = "validation"


# --------------------------------------------------
# Funcoes utilitarias de hiperparametros
# --------------------------------------------------


def _build_folder_name(extractor_name: str, ext_params: dict) -> str:
    """Gera o nome da subpasta a partir dos hiperparametros do extrator."""
    norm = ext_params.get("normalization", "none")
    if extractor_name in ("tfidf", "bow"):
        mf = ext_params["max_features"]
        ng_min, ng_max = ext_params["ngram_range"]
        md = ext_params["min_df"]
        mxd = ext_params["max_df"]
        return f"mf{mf}_ng{ng_min}-{ng_max}_md{md}_mxd{mxd}_norm-{norm}"
    elif extractor_name == "bert":
        ml = ext_params["max_length"]
        mn = ext_params["model_name"]
        return f"ml{ml}_{mn}_norm-{norm}"
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


def _save_params_json(
    features_dir: Path, extractor_name: str, ext_params: dict, metrics: dict = None
):
    """Salva params.json na pasta de features."""
    params_to_save = {"extractor": extractor_name}
    for k, v in ext_params.items():
        # Converte tuplas para listas para serializacao JSON
        if isinstance(v, tuple):
            params_to_save[k] = list(v)
        else:
            params_to_save[k] = v

    if metrics:
        params_to_save["metrics"] = metrics

    with open(features_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(params_to_save, f, indent=2)


def _load_params_json(features_dir: Path) -> dict:
    """Carrega params.json da pasta de features."""
    with open(features_dir / "params.json", "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------
# Funcoes de exportacao de resultados
# --------------------------------------------------


def export_results_to_csv(
    representation: str,
    result_type: ResultType,
    accuracy: float,
    ext_params: dict,
    knn_params: dict,
    metrics: dict = None,
):
    """
    Exporta os resultados do experimento para um arquivo CSV unificado, fazendo append das execucoes.

    Args:
        representation (str): Nome do metodo de extracao de caracteristicas.
        result_type (ResultType): Tipo de resultado (teste ou validacao).
        accuracy (float): Acuracia obtida.
        ext_params (dict): Hiperparametros do extrator.
        knn_params (dict): Hiperparametros do KNN.
        metrics (dict): Dicionario com as metricas de tempo medidas.
    """
    import pandas as pd

    results_dir = PROJECT_ROOT / "results"
    os.makedirs(results_dir, exist_ok=True)

    # Monta a linha de dados combinando tudo
    data = {
        "Extractor": representation,
        "Accuracy": accuracy,
    }

    # Adiciona metricas
    if metrics:
        for k, v in metrics.items():
            data[k] = v

    # Adiciona params do extrator convertendo tuplas/listas para string (ex: ngram_range)
    for k, v in ext_params.items():
        data[k] = str(v) if isinstance(v, (tuple, list)) else v

    # Adiciona params do KNN
    for k, v in knn_params.items():
        data[k] = v

    results_df = pd.DataFrame([data])
    csv_path = results_dir / f"{result_type.value}_results.csv"

    if not csv_path.exists():
        results_df.to_csv(csv_path, index=False)
    else:
        # Le o CSV existente
        existing_df = pd.read_csv(csv_path)
        # Concatena a nova execucao
        combined_df = pd.concat([existing_df, results_df], ignore_index=True)

        # As colunas que definem a unicidade da execucao sao apenas o Extrator e os Hiperparametros
        param_cols = ["Extractor"] + list(ext_params.keys()) + list(knn_params.keys())

        # Remove duplicatas baseadas estritamente nos hiperparametros, mantendo sempre a mais recente ('last')
        combined_df = combined_df.drop_duplicates(subset=param_cols, keep="last")

        # Salva o arquivo atualizado sobrescrevendo o antigo
        combined_df.to_csv(csv_path, index=False)

    print(f"  Resultado salvo/atualizado em: {csv_path}")


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


def _predict_in_batches(model, X, batch_size=500):
    """
    Faz predicoes em lotes (batches) para evitar ArrayMemoryError no KNN.
    Em vez de predizer X inteiro de uma vez, divide em lotes menores.
    """
    import numpy as np

    predictions = []
    num_samples = X.shape[0]

    for i in range(0, num_samples, batch_size):
        X_batch = X[i : i + batch_size]
        pred_batch = model.predict(X_batch)
        predictions.append(pred_batch)

    return np.concatenate(predictions)


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
    t1_extract = time.perf_counter()
    if extractor_name in ("tfidf", "bow"):
        X_train_full, X_test = ext_module.extract(
            X_train_texts,
            X_test_texts,
            max_features=ext_params["max_features"],
            ngram_range=ext_params["ngram_range"],
            min_df=ext_params["min_df"],
            max_df=ext_params["max_df"],
        )
    elif extractor_name == "bert":
        X_train_full, X_test = ext_module.extract(
            X_train_texts,
            X_test_texts,
            max_length=ext_params["max_length"],
            model_name=ext_params["model_name"],
        )
    t2_extract = time.perf_counter()

    print(f"[METRICA] Extracao: {t2_extract - t1_extract:.4f}s")

    # Split estratificado 80/20 para validacao
    print("Fazendo split 80/20 estratificado...")
    y_train_arr = np.array(y_train)

    X_train, X_val, y_train_split, y_val = train_test_split(
        X_train_full,
        y_train_arr,
        test_size=0.20,
        random_state=42,
        stratify=y_train_arr,
    )

    from representation_extraction.dataset.dataset import normalize_features

    print(f"Aplicando normalizacao: {ext_params['normalization']}...")
    X_train, X_val, X_test = normalize_features(
        X_train, X_val, X_test, method=ext_params["normalization"]
    )

    # Salva os 6 arquivos
    print(f"Salvando features em features/{extractor_name}/{folder_name}/...")
    _save_features(extractor_name, feat_dir, "X_train", X_train)
    _save_labels(feat_dir, "y_train", y_train_split)
    _save_features(extractor_name, feat_dir, "X_val", X_val)
    _save_labels(feat_dir, "y_val", y_val)
    _save_features(extractor_name, feat_dir, "X_test", X_test)
    _save_labels(feat_dir, "y_test", np.array(y_test))

    # Salva params.json com a métrica de tempo de extração
    metrics = {"time_extraction_s": t2_extract - t1_extract}
    _save_params_json(feat_dir, extractor_name, ext_params, metrics=metrics)

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
    t1_train = time.perf_counter()
    knn.fit(X_train, y_train)
    t2_train = time.perf_counter()
    print(f"[METRICA] Treinamento KNN: {t2_train - t1_train:.4f}s")

    print("Predizendo no conjunto de validacao em lotes...")
    t1_predict = time.perf_counter()
    predictions = _predict_in_batches(knn, X_val, batch_size=500)
    t2_predict = time.perf_counter()
    print(f"[METRICA] Predicao: {t2_predict - t1_predict:.4f}s")

    pred_dir = PREDICTIONS_DIR / "validation" / extractor_name / folder_name
    os.makedirs(pred_dir, exist_ok=True)

    np.save(pred_dir / "predictions.npy", predictions)

    pred_params = {
        "extractor": extractor_name,
        "ext_params": ext_params,
        "knn_params": knn_params,
        "metrics": {
            "time_extraction_s": _load_params_json(feat_dir)
            .get("metrics", {})
            .get("time_extraction_s", 0.0),
            "time_knn_train_s": t2_train - t1_train,
            "time_knn_predict_s": t2_predict - t1_predict,
        },
    }

    with open(pred_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(pred_params, f, indent=2)

    print(f"  Predicoes e tempos salvos em: {pred_dir}\n")


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
    t1_train = time.perf_counter()
    knn.fit(X_train, y_train)
    t2_train = time.perf_counter()
    print(f"[METRICA] Treinamento KNN: {t2_train - t1_train:.4f}s")

    print("Predizendo no conjunto de teste em lotes...")
    t1_predict = time.perf_counter()
    predictions = _predict_in_batches(knn, X_test, batch_size=500)
    t2_predict = time.perf_counter()
    print(f"[METRICA] Predicao: {t2_predict - t1_predict:.4f}s")
    pred_dir = PREDICTIONS_DIR / "test" / extractor_name / folder_name
    os.makedirs(pred_dir, exist_ok=True)

    np.save(pred_dir / "predictions.npy", predictions)

    pred_params = {
        "extractor": extractor_name,
        "ext_params": ext_params,
        "knn_params": knn_params,
        "metrics": {
            "time_extraction_s": _load_params_json(feat_dir)
            .get("metrics", {})
            .get("time_extraction_s", 0.0),
            "time_knn_train_s": t2_train - t1_train,
            "time_knn_predict_s": t2_predict - t1_predict,
        },
    }

    with open(pred_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(pred_params, f, indent=2)

    print(f"  Predicoes e tempos salvos em: {pred_dir}\n")


def do_stats(extractor_name: str, args):
    """
    Etapa de estatisticas: carrega as predicoes salvas, true labels e hiperparametros.
    Calcula todas as metricas e salva no CSV.
    """
    res_type_str = getattr(args, "result_type", None)
    if not res_type_str:
        raise ValueError("--result-type e obrigatorio para a etapa stats")

    res_type = ResultType(res_type_str)
    ext_params = _get_extractor_params(extractor_name, args)
    folder_name = _build_folder_name(extractor_name, ext_params)

    feat_dir = FEATURES_DIR / extractor_name / folder_name
    pred_dir = PREDICTIONS_DIR / res_type.value / extractor_name / folder_name

    if not pred_dir.exists():
        raise FileNotFoundError(
            f"Pasta de predicoes nao encontrada: {pred_dir}\n"
            f"Execute primeiro a etapa correspondente (validate ou test)."
        )

    print(f"\n{'='*60}")
    print(f"  ESTATISTICAS - {extractor_name.upper()} ({res_type.value})")
    print(f"{'='*60}")

    # Carrega as labels verdadeiras (y_true) a partir do diretorio de features
    if res_type == ResultType.VALIDATION:
        y_true = _load_labels(feat_dir, "y_val")
    else:
        y_true = _load_labels(feat_dir, "y_test")

    y_pred = np.load(pred_dir / "predictions.npy", allow_pickle=True)

    with open(pred_dir / "params.json", "r", encoding="utf-8") as f:
        pred_params = json.load(f)

    time_metrics = pred_params.get("metrics", {})
    # O knn params salvo na predicao eh a fonte da verdade para o deduplicador
    real_knn_params = pred_params.get("knn_params", _get_knn_params(args))

    stats = get_statistics(y_true, y_pred, time_metrics)
    accuracy = stats.pop("Accuracy")

    # Opcional: mostrar logs resumidos
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {stats['Precision']:.4f}")
    print(f"  Recall: {stats['Recall']:.4f}")
    print(f"  F1_Score: {stats['F1_Score']:.4f}")

    export_results_to_csv(
        representation=extractor_name,
        result_type=res_type,
        accuracy=accuracy,
        ext_params=ext_params,
        knn_params=real_knn_params,
        metrics=stats,
    )


def do_gridsearch(extractor_name: str, args):
    """
    Etapa de Grid Search: usa Pipeline (extrator + KNN) com GridSearchCV do sklearn
    para buscar exaustivamente todas as combinacoes de hiperparametros.
    O sklearn cuida de todas as permutacoes e do 5-fold CV internamente.
    Salva os resultados em results/{extractor}/gridsearch_results.csv.
    """
    import pandas as pd
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from nltk.corpus import stopwords
    import nltk

    if extractor_name == "all":
        raise ValueError(
            "Grid search deve ser executado para um extrator especifico, nao 'all'."
        )

    param_grid = GRIDSEARCH_PARAM_GRIDS[extractor_name]

    # Monta o Pipeline de acordo com o extrator
    try:
        stop_words_en = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")
        stop_words_en = stopwords.words("english")

    if extractor_name == "bow":
        extractor_step = CountVectorizer(stop_words=stop_words_en, lowercase=True)
    elif extractor_name == "tfidf":
        extractor_step = TfidfVectorizer(stop_words=stop_words_en, lowercase=True)
    elif extractor_name == "bert":
        raise NotImplementedError(
            "Grid search com Pipeline nao suportado para BERT (requer transformer wrapper)."
        )

    # Monta pipeline com passo 'scaler' configurado como 'passthrough' por padrao
    # O GridSearchCV se encarregara de testar os outros definidos no param_grid
    steps = [
        ("extractor", extractor_step),
        ("scaler", "passthrough"),
        ("knn", KNeighborsClassifier()),
    ]
    pipe = Pipeline(steps)

    # Contagem de combinacoes para log
    from sklearn.model_selection import ParameterGrid

    total_combos = len(ParameterGrid(param_grid))

    print(f"\n{'='*60}")
    print(f"  GRID SEARCH - {extractor_name.upper()}")
    print(f"  Pipeline: {' -> '.join(name for name, _ in steps)}")
    print(f"  Total de combinacoes: {total_combos}")
    print(f"  Folds: 5 (StratifiedKFold)")
    print(f"  Total de fits: {total_combos * 5}")
    print(f"{'='*60}\n")

    # Carrega dataset
    print("Carregando dataset...")
    X_train_texts, y_train = load_dataset("comments_train.txt")
    y_train_arr = np.array(y_train)

    # GridSearchCV cuida de TUDO: permutacoes, folds, fit, score
    grid_cv = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
        verbose=3,
        return_train_score=False,
    )

    print("Rodando GridSearchCV...")
    t1 = time.perf_counter()
    grid_cv.fit(X_train_texts, y_train_arr)
    t2 = time.perf_counter()

    print(f"\nGridSearchCV concluido em {t2 - t1:.2f}s")
    print(f"Melhor accuracy: {grid_cv.best_score_:.4f}")
    print(f"Melhores params: {grid_cv.best_params_}")

    # Converte cv_results_ diretamente para DataFrame e salva
    results_dir = PROJECT_ROOT / "results" / extractor_name
    os.makedirs(results_dir, exist_ok=True)
    csv_path = results_dir / "gridsearch_results.csv"

    results_df = pd.DataFrame(grid_cv.cv_results_)
    results_df = results_df.sort_values("rank_test_score")
    results_df.to_csv(csv_path, index=False)

    print(f"\n{'='*60}")
    print(f"  GRID SEARCH CONCLUIDO - {extractor_name.upper()}")
    print(f"  Total de combinacoes avaliadas: {total_combos}")
    print(f"  Resultados salvos em: {csv_path}")
    print(f"  Top 5 resultados:")
    top5 = results_df.head(5)
    for idx, row in top5.iterrows():
        print(
            f"    #{int(row['rank_test_score'])}: accuracy={row['mean_test_score']:.4f} (+/-{row['std_test_score']:.4f}) | {row['params']}"
        )
    print(f"{'='*60}\n")


def do_all(extractor_name: str, args):
    """Roda extract -> validate -> stats(val) -> test -> stats(test) em sequencia."""
    do_extract(extractor_name, args)

    do_validate(extractor_name, args)
    args.result_type = "validation"
    do_stats(extractor_name, args)

    do_test(extractor_name, args)
    args.result_type = "test"
    do_stats(extractor_name, args)


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
        "--max-features",
        type=int,
        default=None,
        help="Numero maximo de features (bow/tfidf, default: 350)",
    )
    parser.add_argument(
        "--ngram-range",
        type=int,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="Range de n-grams (bow/tfidf, default: 1 2)",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=None,
        help="Frequencia minima de documento (bow/tfidf, default: 2)",
    )
    parser.add_argument(
        "--max-df",
        type=float,
        default=None,
        help="Frequencia maxima de documento (bow/tfidf, default: 0.9)",
    )
    # BERT
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Comprimento maximo da sequencia de tokens (bert, default: 16)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Nome do modelo HuggingFace (bert, default: bert-base-uncased)",
    )
    # Global
    parser.add_argument(
        "--normalization",
        type=str,
        default=None,
        choices=["none", "maxabs", "normalizer"],
        help="Metodo de normalizacao (default: none)",
    )


def _add_knn_args(parser):
    """Adiciona argumentos de hiperparametros do KNN ao parser."""
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=None,
        help="Numero de vizinhos do KNN (default: 7)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        help="Metrica de distancia do KNN (default: euclidean)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        choices=["uniform", "distance"],
        help="Peso dos vizinhos do KNN (default: uniform)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline de extracao, validacao, teste e avaliacao de representacoes textuais.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Exemplos de uso:
  python compare.py extract  --extractor tfidf
  python compare.py validate --extractor tfidf --n-neighbors 5
  python compare.py stats    --extractor tfidf --result-type validation
  python compare.py test     --extractor bow
  python compare.py all      --extractor all
""",
    )

    subparsers = parser.add_subparsers(dest="stage", required=True)

    extractor_choices = list(EXTRACTORS.keys()) + ["all"]

    # extract: hiper. do extrator
    sub_extract = subparsers.add_parser(
        "extract", help="Extrai features e salva em disco"
    )
    sub_extract.add_argument(
        "--extractor",
        required=True,
        choices=extractor_choices,
        help="Extrator a usar (ou 'all' para todos)",
    )
    _add_extractor_args(sub_extract)

    # validate: hiper. do extrator (para localizar pasta) + hiper. do KNN
    sub_validate = subparsers.add_parser(
        "validate", help="Valida usando features pre-extraidas (split 80/20)"
    )
    sub_validate.add_argument(
        "--extractor",
        required=True,
        choices=extractor_choices,
        help="Extrator a usar (ou 'all' para todos)",
    )
    _add_extractor_args(sub_validate)
    _add_knn_args(sub_validate)

    # test: hiper. do extrator (para localizar pasta) + hiper. do KNN
    sub_test = subparsers.add_parser("test", help="Testa usando features pre-extraidas")
    sub_test.add_argument(
        "--extractor",
        required=True,
        choices=extractor_choices,
        help="Extrator a usar (ou 'all' para todos)",
    )
    _add_extractor_args(sub_test)
    _add_knn_args(sub_test)

    # stats: hiper. do extrator para achar a pasta e result type
    sub_stats = subparsers.add_parser(
        "stats", help="Gera as estatisticas usando predicoes salvas"
    )
    sub_stats.add_argument(
        "--extractor",
        required=True,
        choices=extractor_choices,
        help="Extrator a usar (ou 'all' para todos)",
    )
    sub_stats.add_argument(
        "--result-type",
        type=str,
        choices=["validation", "test"],
        help="Qual predicao analisar (validation ou test)",
    )
    _add_extractor_args(sub_stats)

    # gridsearch: apenas --extractor e --normalization, grids hardcoded
    gridsearch_choices = list(EXTRACTORS.keys())  # sem 'all'
    sub_gridsearch = subparsers.add_parser(
        "gridsearch",
        help="Executa grid search exaustivo (5-fold CV) sobre hiperparametros",
    )
    sub_gridsearch.add_argument(
        "--extractor",
        required=True,
        choices=gridsearch_choices,
        help="Extrator a usar (sem 'all')",
    )

    # all: hiper. do extrator + hiper. do KNN
    sub_all = subparsers.add_parser(
        "all", help="Roda extract -> validate -> stats -> test -> stats em sequencia"
    )
    sub_all.add_argument(
        "--extractor",
        required=True,
        choices=extractor_choices,
        help="Extrator a usar (ou 'all' para todos)",
    )
    _add_extractor_args(sub_all)
    _add_knn_args(sub_all)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Valida hiperparametros do extrator
    if args.extractor != "all" and args.stage not in ("gridsearch",):
        _validate_extractor_args(args.extractor, args)

    stage_map = {
        "extract": do_extract,
        "validate": do_validate,
        "test": do_test,
        "stats": do_stats,
        "gridsearch": do_gridsearch,
        "all": do_all,
    }

    stage_fn = stage_map[args.stage]
    _run_stage(stage_fn, args.extractor, args)


if __name__ == "__main__":
    main()
