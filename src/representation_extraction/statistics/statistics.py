from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import pandas as pd
from pathlib import Path
import os
from representation_extraction.constants import ResultType

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def get_statistics(y_true, y_pred, time_metrics: dict) -> dict:
    """
    Calcula estatisticas de desempenho do classificador.
    Retorna as metricas de classificacao somadas as metricas de tempo extraidas do dict.
    """
    accuracy = accuracy_score(y_true, y_pred)

    # average='macro' calcula as metricas independentemente para cada label
    # e encontra a media nao ponderada, lidando bem com desbalanceamento de classes sem vies.
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)

    stats = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1_score,
        "Confusion_Matrix": cm.tolist(),
    }

    if time_metrics:
        stats.update(time_metrics)

    return stats


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
