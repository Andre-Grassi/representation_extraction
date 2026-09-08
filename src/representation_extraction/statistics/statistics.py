from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

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
