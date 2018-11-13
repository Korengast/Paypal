__author__ = "Koren Gast"
from models.model import GenModel
from sklearn.neural_network import MLPClassifier


class MLP(GenModel):
    def __init__(self):
        super().__init__()
        self.model = MLPClassifier(hidden_layer_sizes=[128, 64, ])