import pandas as pd


def load_dataset(txt_path: str) -> tuple[pd.Series, pd.Series]:
    """
    Carrega um dataset a partir de um arquivo de texto no formato csv.

    Args:
        txt_path (str): Caminho para o arquivo de texto contendo o dataset.

    Returns:
        tuple[list[str], list[str]]: Uma tupla contendo duas listas:
            - A primeira lista contém os textos (reviews) do dataset.
            - A segunda lista contém os rótulos (labels) correspondentes aos textos.
    """
    df = pd.read_csv(txt_path)

    texts = df["review"].astype(str)
    labels = df["label"]

    return (texts, labels)
