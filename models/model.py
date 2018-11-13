__author__ = "Koren Gast"
from sklearn.metrics import accuracy_score

class GenModel(object):
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        self.model.fit(X, y)

    def evaluate(self, X, y):
        preds = self.predict(X)
        return accuracy_score(y, preds)

    def predict(self, X):
        return self.model.predict(X)
