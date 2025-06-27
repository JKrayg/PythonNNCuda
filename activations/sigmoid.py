import cupy as cp

from activations.activation import Activation
from initializers.glorot import Glorot
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from layers.layer import Layer


class Sigmoid(Activation):
    def __init__(self):
        print("sigmoid")
    
    def initWB(self, prev: "Layer", curr: "Layer"):
        w = Glorot.initWeights(prev, curr)
        b = cp.zeros((curr.numNeurons, 1))

        return w, b