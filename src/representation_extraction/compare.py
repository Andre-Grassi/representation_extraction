from pathlib import Path

import pandas as pd
import os
from enum import Enum
import argparse

from sklearn.neighbors import KNeighborsClassifier
from representation_extraction.extractors import tfidf, bow, bert

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Enum para indicar se é teste ou validação
class ResultType(Enum):
    TEST = "test"
    VALIDATION = "validation"


def export_results_to_csv(
    representation: str,
    result_type: ResultType,
    accuracy_validation: float,
    accuracy_test: float,
    execution_time: float,
):
    """
    Exporta os resultados do experimento para um arquivo CSV na pasta 'results/{result_type}/{representation}.csv'.

    Args:
        representation (str): Nome do método de extração de características.
        accuracy_validation (float): Acurácia obtida na validação.
        accuracy_test (float): Acurácia obtida no teste.
        execution_time (float): Tempo de execução do experimento em segundos.
    """

    # Cria a pasta 'results/' se não existir
    os.makedirs(PROJECT_ROOT / "results", exist_ok=True)

    # Cria a pasta 'results/test' ou 'results/validation' dependendo do tipo de resultado
    if result_type == ResultType.TEST:
        os.makedirs(PROJECT_ROOT / "results/test", exist_ok=True)
    elif result_type == ResultType.VALIDATION:
        os.makedirs(PROJECT_ROOT / "results/validation", exist_ok=True)

    # Cria o DataFrame com os resultados
    results_df = pd.DataFrame(
        {
            "Representation": [representation],
            "Accuracy_Validation": [accuracy_validation],
            "Accuracy_Test": [accuracy_test],
            "Execution_Time": [execution_time],
        }
    )

    # Define o caminho do arquivo CSV
    if result_type == ResultType.TEST:
        csv_path = os.path.join(PROJECT_ROOT / "results/test", f"{representation}.csv")
    elif result_type == ResultType.VALIDATION:
        csv_path = os.path.join(
            PROJECT_ROOT / "results/validation", f"{representation}.csv"
        )
    else:
        raise ValueError("Invalid result_type. Must be 'test' or 'validation'.")

    # Exporta para CSV
    results_df.to_csv(csv_path, index=False)


def run_tfidf(knn):
    test_accuracy = tfidf.run(knn)

    export_results_to_csv(
        representation="tfidf",
        result_type=ResultType.TEST,
        accuracy_validation=0.0,
        accuracy_test=test_accuracy,
        execution_time=0.0,
    )
    print("Acurácia do TESTE: {:.4f}".format(test_accuracy))


def run_bow(knn):

    test_accuracy = bow.run(knn)

    export_results_to_csv(
        representation="bow",
        result_type=ResultType.TEST,
        accuracy_validation=0.0,
        accuracy_test=test_accuracy,
        execution_time=0.0,
    )
    print("Acurácia do TESTE: {:.4f}".format(test_accuracy))


def run_bert(knn):
    test_accuracy = bert.run(knn)

    export_results_to_csv(
        representation="bert",
        result_type=ResultType.TEST,
        accuracy_validation=0.0,
        accuracy_test=test_accuracy,
        execution_time=0.0,
    )
    print("Acurácia do TESTE: {:.4f}".format(test_accuracy))


def main():
    # Cria knn
    knn = KNeighborsClassifier(n_neighbors=7, metric="euclidean")

    run_bert(knn)


if __name__ == "__main__":
    main()
