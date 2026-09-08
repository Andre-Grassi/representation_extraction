import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = PROJECT_ROOT / "dataset"


def load_dataset(file_name: str) -> tuple[pd.Series, pd.Series]:
    """
    Carrega um dataset a partir de um arquivo de texto no formato csv.

    Args:
        txt_path (str): Caminho para o arquivo de texto contendo o dataset.

    Returns:
        tuple[pd.Series, pd.Series]: Uma tupla contendo duas séries do pandas:
            - A primeira série contém os textos (reviews).
            - A segunda série contém os rótulos (labels).
    """
    # Se um caminho abosluto for fornecido, use-o diretamente; caso contrário, construa o caminho relativo ao diretório do dataset
    file_path = Path(file_name)
    if not file_path.is_absolute():
        file_path = DATASET_DIR / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"O arquivo {file_path} não foi encontrado.")

    df = pd.read_csv(file_path)

    texts = df["review"].astype(str)
    labels = df["label"]

    return (texts, labels)
