import math
import cupy as cp

from activations.activation import Activation
from initializers.glorot import Glorot
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from layers.layer import Layer


class Sigmoid(Activation):
    def __init__(self):
        print("sigmoid")

    def activate(z: cp.ndarray):
        return 1 / (1 + cp.exp(-z))
    
    def initWB(rows: int, cols: int):
        w = Glorot.initWeights(rows, cols)
        b = cp.zeros((cols))

        return w, b