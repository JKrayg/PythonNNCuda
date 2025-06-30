from activations.activation import Activation
from initializers.he import He
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from layers.layer import Layer

import cupy as cp

class ReLU(Activation):
    def __init__(self):
        print("relu")

    def activate(z: cp.ndarray):
        return (z > 0).astype(cp.float32)
    
    def gradient(preAct: cp.ndarray, gradPreAct: cp.ndarray):
        der = (preAct > 0).astype(cp.float32)
        return der * gradPreAct

    def initWB(rows: int, cols: int):
        w = He.initWeights(rows, cols)
        b = cp.full((cols), 0.1)

        return w, b