import cupy as cp;
import numpy as np

from typing import List, Optional
from layers.layer import Layer;
from activations.activation import Activation
from training.loss.loss import Loss
from utils.utils import Utils

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
        self.preactivation = cp.empty((batchSize, self.numNeurons))

        if prev != None:
            w, b = actFunc.initWB(prev.numNeurons, self.numNeurons)
            self.weights = w
            self.bias = b
        else:
            w, b = actFunc.initWB(self.numFeatures, self.numNeurons)
            self.weights = w
            self.bias = b


        if self.normalizer != None:
            scVar = cp.fill((self.numNeurons, 1), 1)
            shMeans = cp.ndarray((self.numNeurons, 1))
            self.normalizer.scale = scVar
            self.normalizer.shift = shMeans
            self.normalizer.means = shMeans
            self.normalizer.variances = scVar
            self.normalizer.runMeans = shMeans
            self.normalizer.runVar = scVar

        return self
    

    def forwardProp(self, prev: Layer):
        maths = Utils()
        z = maths.weightedSum(prev.activation, self)
        self.preactivation = z

        if (self.normalizer != None):
            z = self.normalizer.normalize(z)

        activated = None
        if (self.actFunc != None):
            activated = self.actFunc.execute(z)
        else:
            activated = z

        if (self.regularizers != None):
            for r in self.regularizers:
                activated = r.regularize(activated)



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