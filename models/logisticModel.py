__author__ = "Koren Gast"
from models.model import GenModel
from sklearn import linear_model


class LogisticModel(GenModel):
    def __init__(self):
        super().__init__()
        self.model = linear_model.LogisticRegression()
