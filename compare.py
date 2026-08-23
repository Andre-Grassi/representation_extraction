import pandas as pd
import os
from enum import Enum


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
    os.makedirs("results", exist_ok=True)

    # Cria a pasta 'results/test' ou 'results/validation' dependendo do tipo de resultado
    if result_type == ResultType.TEST:
        os.makedirs("results/test", exist_ok=True)
    elif result_type == ResultType.VALIDATION:
        os.makedirs("results/validation", exist_ok=True)

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
        csv_path = os.path.join("results/test", f"{representation}.csv")
    elif result_type == ResultType.VALIDATION:
        csv_path = os.path.join("results/validation", f"{representation}.csv")
    else:
        raise ValueError("Invalid result_type. Must be 'test' or 'validation'.")

    # Exporta para CSV
    results_df.to_csv(csv_path, index=False)
