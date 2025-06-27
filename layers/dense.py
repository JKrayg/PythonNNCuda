import cupy as cp;
import numpy as np

from typing import List, Optional
from layers.layer import Layer;
from activations.activation import Activation
from training.loss.loss import Loss

class Dense(Layer):
    def __init__(self, numNeurons: int, actFunc: Activation, inputShape: Optional[List[int]] = None, lossFunc: Loss = None):
        biases = cp.ndarray(numNeurons)
        super().__init__(actFunc, inputShape)
        self.numNeurons = numNeurons
        self.lossFunc = lossFunc
        self.numFeatures = inputShape





    def initLayer(self, prev: Layer, batchSize: int):
        actFunc: Activation = self.actFunc
        self.activation = cp.empty((batchSize, self.numNeurons))

        if prev != None:
            w, b = actFunc.initWB(prev, self)
            self.weights = w
            self.bias = b
        else:
            w, b = actFunc.initWB(self.numFeatures, self)
            self.weights = w
            self.bias = b







    def adamInit(self):
        w = cp.ndarray(self.weights[0], self.weights[1])
        b = cp.ndarray(self.bias[0], self.bias[1])

        self.weightsMomentum(w)
        self.weightsVariance(w)
        self.biasMomentum(b)
        self.biasVariance(b)

        norm = self.normalizers

        if (norm != None):
            shift = cp.ndarray(norm.shift[0], norm.shift[1])
            scale = cp.ndarray(norm.scale[0], norm.scale[1])

            norm.shiftMomentum = shift
            norm.shiftVariance = shift
            norm.scaleMomentum = scale
            norm.scaleVariance = scale