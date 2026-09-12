from sklearn.neighbors import KNeighborsClassifier
import numpy as np


class BatchedKNeighborsClassifier(KNeighborsClassifier):
    """
    Extensão do KNeighborsClassifier do scikit-learn que realiza
    predições em lotes (batches) para evitar estouro de memória.
    """

    def __init__(
        self,
        n_neighbors=7,
        weights="uniform",
        metric="euclidean",
        batch_size=500,
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
        )
        self.batch_size = batch_size

    def _predict_in_batches(self, X, method):
        """Executa a função de predição dividindo os dados em lotes."""
        n_samples = X.shape[0]
        predictions = []
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            X_batch = X[start_idx:end_idx]
            batch_preds = method(X_batch)
            predictions.append(batch_preds)
        return np.concatenate(predictions, axis=0)

    def predict(self, X):
        return self._predict_in_batches(X, super().predict)

    def predict_proba(self, X):
        return self._predict_in_batches(X, super().predict_proba)
