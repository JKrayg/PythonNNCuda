import cupy as cp
from activations.activation import Activation
from typing import TYPE_CHECKING

from initializers.glorot import Glorot

if TYPE_CHECKING:
    from layers.layer import Layer


class Tanh(Activation):
    def __init__(self):
        print("tanh")

    def initWB(self, rows: int, cols: int):
        w = Glorot.initWeights(rows, cols)
        b = cp.zeros((cols))

        return w, b
    
    # def initWB(self, inputSize: int, curr: "Layer"):
    #     w = Glorot.initWeights(inputSize, curr)
    #     b = cp.zeros((curr.numNeurons, 1))

    #     return w, b