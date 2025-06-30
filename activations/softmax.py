import cupy as cp

from activations.activation import Activation
from initializers.glorot import Glorot
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from layers.layer import Layer


class Softmax(Activation):
    def __init__(self):
        print("sofmax")

    def initWB(self, rows: int, cols: int):
        w = Glorot.initWeights(rows, cols)
        b = cp.zeros((cols))

        return w, b
    
    # def initWB(self, inputSize: int, curr: "Layer"):
    #     w = Glorot.initWeights(inputSize, curr)
    #     b = cp.zeros((curr.numNeurons, 1))

    #     return w, b