import cupy as cp

from layers.layer import Layer

class Flatten(Layer):
    def __init__(self):
        self.previousShape = None

    def reshapeGrad(self, grad: cp.ndarray):
        return grad.reshape(self.previousShape)