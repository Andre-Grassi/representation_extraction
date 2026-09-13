from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin

class BaseExtractor(ABC, BaseEstimator, TransformerMixin):
    """
    Classe base abstrata para todos os extratores de representação.
    Garante a interface do scikit-learn (fit, transform).
    """

    @abstractmethod
    def fit(self, X, y=None):
        """Treina o extrator com os textos."""
        pass

    @abstractmethod
    def transform(self, X):
        """Transforma os textos em representações."""
        pass

    def extract(self, X_train_texts, X_test_texts):
        """
        Método utilitário para extrair features de treino e teste de uma vez.
        Faz o fit no treino e transforma ambos.
        """
        self.fit(X_train_texts)
        X_train = self.transform(X_train_texts)
        X_test = self.transform(X_test_texts)
        return X_train, X_test
