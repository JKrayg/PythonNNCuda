import cupy as cp

from activations.activation import Activation
from initializers.glorot import Glorot
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from layers.layer import Layer


class Linear(Activation):
    def __init__(self):
        print("linear")
    
    def initWB(self, rows: int, cols: int):
        w = Glorot.initWeights(rows, cols)
        b = cp.zeros((cols))

        return w, b