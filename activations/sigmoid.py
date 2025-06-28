import cupy as cp

from activations.activation import Activation
from initializers.glorot import Glorot
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from layers.layer import Layer


class Sigmoid(Activation):
    def __init__(self):
        print("sigmoid")
    
    def initWB(self, rows: int, cols: int):
        w = Glorot.initWeights(rows, cols)
        b = cp.zeros((cols, 1))

        return w, b