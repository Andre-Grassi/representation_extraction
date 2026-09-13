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
from representation_extraction.knn.knn import BatchedKNeighborsClassifier


from representation_extraction.statistics.statistics import (
    export_results_to_csv,
    get_statistics,
)


from representation_extraction.constants import (
    PROJECT_ROOT,
    FEATURES_DIR,
    PREDICTIONS_DIR,
    EXTRACTORS,
    EXTRACTOR_PARAMS,
    EXTRACTOR_DEFAULTS,
    KNN_DEFAULTS,
    GRIDSEARCH_PARAM_GRIDS,
    ResultType,
)
from representation_extraction.parse import (
    build_folder_name,
    get_extractor_params,
    get_knn_params,
    validate_extractor_args,
)

# --------------------------------------------------
# Funcoes utilitarias de hiperparametros
# --------------------------------------------------


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
    ext_params = get_extractor_params(extractor_name, args)
    folder_name = build_folder_name(extractor_name, ext_params)
    feat_dir = FEATURES_DIR / extractor_name / folder_name

    print(f"\n{'='*60}")
    print(f"  EXTRACAO - {extractor_name.upper()}")
    print(f"  Pasta: features/{extractor_name}/{folder_name}/")
    print(f"  Params: {ext_params}")
    print(f"{'='*60}")

    ExtractorClass = EXTRACTORS[extractor_name]["class"]

    # Carrega dataset
    print("Carregando dataset...")
    X_train_texts, y_train = load_dataset("comments_train.txt")
    X_test_texts, y_test = load_dataset("comments_test.txt")

    # Para BERT, precisa converter para lista
    if extractor_name == "bert":
        X_train_texts = X_train_texts.tolist()
        X_test_texts = X_test_texts.tolist()

    # Instancia o extrator com os parâmetros
    extractor_instance = ExtractorClass(**ext_params)

    # Extrai features
    print(f"Extraindo features com {extractor_name}...")
    t1_extract = time.perf_counter()
    X_train_full, X_test = extractor_instance.extract(X_train_texts, X_test_texts)
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
    ext_params = get_extractor_params(extractor_name, args)
    knn_params = get_knn_params(args)
    folder_name = build_folder_name(extractor_name, ext_params)
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
    ext_params = get_extractor_params(extractor_name, args)
    knn_params = get_knn_params(args)
    folder_name = build_folder_name(extractor_name, ext_params)
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
    ext_params = get_extractor_params(extractor_name, args)
    folder_name = build_folder_name(extractor_name, ext_params)

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
    real_knn_params = pred_params.get("knn_params", get_knn_params(args))

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
    Etapa de Grid Search: busca exaustivamente todas as combinacoes de hiperparametros.

    Para BoW/TF-IDF: usa Pipeline (extrator + scaler + KNN) com GridSearchCV.
    Para BERT: itera manualmente sobre as combinacoes do extrator (poucas),
               extrai features UMA VEZ por combinacao, e roda GridSearchCV
               apenas nos params do KNN + scaler (evita OOM por re-carregar o modelo).

    Salva os resultados em results/{extractor}/gridsearch_results.csv.
    """
    import pandas as pd
    from sklearn.model_selection import GridSearchCV, ParameterGrid
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from nltk.corpus import stopwords
    import nltk

    if extractor_name == "all":
        raise ValueError(
            "Grid search deve ser executado para um extrator especifico, nao 'all'."
        )

    param_grid = GRIDSEARCH_PARAM_GRIDS[extractor_name]
    n_jobs = getattr(args, "n_jobs", 1)

    # Carrega dataset
    print("Carregando dataset...")
    X_train_texts, y_train = load_dataset("comments_train.txt")
    y_train_arr = np.array(y_train)

    results_dir = PROJECT_ROOT / "results" / extractor_name
    os.makedirs(results_dir, exist_ok=True)
    csv_path = results_dir / "gridsearch_results.csv"

    if extractor_name in ("bow", "tfidf"):
        # --------------------------------------------------
        # BoW / TF-IDF: Pipeline completo com GridSearchCV
        # --------------------------------------------------
        ExtractorClass = EXTRACTORS[extractor_name]["class"]
        extractor_step = ExtractorClass()

        steps = [
            ("extractor", extractor_step),
            ("scaler", "passthrough"),
            ("knn", BatchedKNeighborsClassifier()),
        ]
        pipe = Pipeline(steps)

        total_combos = len(ParameterGrid(param_grid))
        print(f"\n{'='*60}")
        print(f"  GRID SEARCH - {extractor_name.upper()}")
        print(f"  Pipeline: {' -> '.join(name for name, _ in steps)}")
        print(f"  Total de combinacoes: {total_combos}")
        print(f"  Folds: 3 | n_jobs: {n_jobs}")
        print(f"  Total de fits: {total_combos * 3}")
        print(f"{'='*60}\n")

        grid_cv = GridSearchCV(
            pipe,
            param_grid=param_grid,
            scoring="accuracy",
            cv=3,
            n_jobs=n_jobs,
            verbose=1,
            return_train_score=False,
        )

        print("Rodando GridSearchCV...")
        t1 = time.perf_counter()
        grid_cv.fit(X_train_texts, y_train_arr)
        t2 = time.perf_counter()

        print(f"\nGridSearchCV concluido em {t2 - t1:.2f}s")
        print(f"Melhor accuracy: {grid_cv.best_score_:.4f}")
        print(f"Melhores params: {grid_cv.best_params_}")

        results_df = pd.DataFrame(grid_cv.cv_results_)
        results_df = results_df.sort_values("rank_test_score")
        results_df.to_csv(csv_path, index=False)

    elif extractor_name == "bert":
        # --------------------------------------------------
        # BERT: loop manual sobre params do extrator,
        # GridSearchCV apenas sobre KNN + scaler
        # --------------------------------------------------

        # Separa params do extrator vs params do KNN+scaler
        ext_param_grid = {
            k.replace("extractor__", ""): v
            for k, v in param_grid.items()
            if k.startswith("extractor__")
        }
        knn_scaler_grid = {
            k: v for k, v in param_grid.items() if not k.startswith("extractor__")
        }
        # Remove prefixo knn__ para o GridSearchCV direto no KNN
        knn_scaler_grid_clean = {}
        for k, v in knn_scaler_grid.items():
            if k.startswith("knn__"):
                knn_scaler_grid_clean[k.replace("knn__", "")] = v
            else:
                knn_scaler_grid_clean[k] = v

        ext_combos = list(ParameterGrid(ext_param_grid))
        knn_combos_count = len(ParameterGrid(knn_scaler_grid_clean))

        print(f"\n{'='*60}")
        print(f"  GRID SEARCH - BERT (modo otimizado)")
        print(f"  Combinacoes do extrator: {len(ext_combos)}")
        print(f"  Combinacoes KNN+scaler por extrator: {knn_combos_count}")
        print(f"  Total: {len(ext_combos) * knn_combos_count}")
        print(f"  Folds: 3 | n_jobs: {n_jobs}")
        print(f"{'='*60}\n")

        X_train_list = (
            X_train_texts.tolist()
            if hasattr(X_train_texts, "tolist")
            else list(X_train_texts)
        )

        all_results = []

        for i, ext_params in enumerate(ext_combos, 1):
            print(f"\n--- Extrator combo {i}/{len(ext_combos)}: {ext_params} ---")

            # Extrai features UMA VEZ para esta combinacao
            ExtractorClass = EXTRACTORS["bert"]["class"]
            transformer = ExtractorClass(**ext_params)
            t1_ext = time.perf_counter()
            transformer.fit(X_train_list)
            X_features = transformer.transform(X_train_list)
            t2_ext = time.perf_counter()
            print(f"  Extracao: {t2_ext - t1_ext:.2f}s | Shape: {X_features.shape}")

            # Para a busca no scaler, precisamos de Pipeline(scaler, knn)
            mini_pipe = Pipeline(
                [
                    ("scaler", "passthrough"),
                    ("knn", BatchedKNeighborsClassifier()),
                ]
            )

            # Ajusta prefixos para o mini pipeline
            mini_param_grid = {}
            for k, v in knn_scaler_grid.items():
                if k.startswith("knn__"):
                    mini_param_grid[k] = v
                else:
                    mini_param_grid[k] = v

            grid_cv = GridSearchCV(
                mini_pipe,
                param_grid=mini_param_grid,
                scoring="accuracy",
                cv=3,
                n_jobs=n_jobs,
                verbose=3,
                return_train_score=False,
            )

            print(f"  Rodando GridSearchCV (KNN + scaler)...")
            t1_grid = time.perf_counter()
            grid_cv.fit(X_features, y_train_arr)
            t2_grid = time.perf_counter()
            print(
                f"  Concluido em {t2_grid - t1_grid:.2f}s | Melhor: {grid_cv.best_score_:.4f}"
            )

            # Coleta resultados e adiciona os params do extrator
            cv_df = pd.DataFrame(grid_cv.cv_results_)
            for k, v in ext_params.items():
                cv_df[f"param_extractor__{k}"] = (
                    str(v) if isinstance(v, (tuple, list)) else v
                )
            cv_df["time_extraction_s"] = t2_ext - t1_ext
            all_results.append(cv_df)

            # Libera memoria do modelo BERT
            del transformer
            import gc

            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        # Consolida todos os resultados
        results_df = pd.concat(all_results, ignore_index=True)
        results_df = results_df.sort_values("mean_test_score", ascending=False)
        results_df.insert(0, "rank_global", range(1, len(results_df) + 1))
        results_df.to_csv(csv_path, index=False)

    print(f"\n{'='*60}")
    print(f"  GRID SEARCH CONCLUIDO - {extractor_name.upper()}")
    print(f"  Resultados salvos em: {csv_path}")
    print(f"  Top 5 resultados:")
    top5 = results_df.head(5)
    for idx, row in top5.iterrows():
        score = row["mean_test_score"]
        std = row["std_test_score"]
        params = {
            k.replace("param_", ""): row[k] for k in row.index if k.startswith("param_")
        }
        print(f"    accuracy={score:.4f} (+/-{std:.4f}) | {params}")
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
        help="Executa grid search exaustivo (3-fold CV) sobre hiperparametros",
    )
    sub_gridsearch.add_argument(
        "--extractor",
        required=True,
        choices=gridsearch_choices,
        help="Extrator a usar (sem 'all')",
    )
    sub_gridsearch.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Numero de workers em paralelo para o GridSearchCV (default: 1, use -1 para todos os cores)",
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
        validate_extractor_args(args.extractor, args)

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
