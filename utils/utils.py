import cupy as cp

from layers.layer import Layer

class Utils:
    def __inti__():
        print("utils")

    def weightedSum(self, activation: cp.ndarray, layer: Layer):
        return activation @ layer.weights + layer.bias
    