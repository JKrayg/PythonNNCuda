import cupy as cp

from layers.layer import Layer

class Utils:
    def __inti__():
        print("utils")

    def weightedSum(self, activation: cp.ndarray, layer: Layer):
        # print("self:", layer.numNeurons)
        # print("----", activation.shape)
        # print("_____", layer.weights.shape)
        # print("+++++", layer.bias.shape)
        return activation @ layer.weights + layer.bias