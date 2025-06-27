from activations.activation import Activation
from initializers.he import He
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from layers.layer import Layer

import cupy as cp

class ReLU(Activation):
    def __init__(self):
        print("relu")

    def initWB(self, prev: "Layer", curr: "Layer"):
        w = He.initWeights(prev, curr)
        b = cp.full((curr.numNeurons, 1), 0.1)

        return w, b