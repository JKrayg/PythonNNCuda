import cupy as cp

from activations.activation import Activation
from initializers.glorot import Glorot
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from layers.layer import Layer


class Softmax(Activation):
    def __init__(self):
        print("sofmax")

    def activate(z: cp.ndarray):
        max = cp.max(z, axis=1, keepdims=True)
        expZ = cp.exp(z - max)
        sumExpZ = cp.sum(expZ, axis=1, keepdims=True)

        return expZ / sumExpZ

    def initWB(rows: int, cols: int):
        w = Glorot.initWeights(rows, cols)
        b = cp.zeros((cols))

        return w, b