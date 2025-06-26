from typing import List, Optional
import cupy as cp;

from activations.activation import Activation
from training.loss.loss import Loss
from training.optimizers.optimizer import Optimizer

class Layer:
    def __init__(self, actFunc: Activation, inputShape: Optional[List[int]] = None):
        self.actFunc = actFunc
        self.inputShape = inputShape
        self.preactivation = None
        self.activation = None
        self.weights = None
        self.weightsMomentum = None
        self.weightsVariance = None
        self.bias = None
        self.biasMomentum = None
        self.biasVariance = None
        self.gradientWrtWeights = None
        self.gradientWrtBias = None
        self.regularizers = None
        self.normalizers = None

    def gradientWeights(self, activation: cp.ndarray, gradient: cp.ndarray):
        gWrTw = cp.transpose(activation) @ gradient / activation.shape[0]
        return gWrTw
    
    def gradientBias(self, gradient: cp.ndarray):
        sum_ = gradient.sum(axis=0).reshape(gradient.shape[1], 1) / gradient.shape[0]
        return sum_
    
    def updateBias(self, o: Optimizer):
        self.bias = o.biasUpdate(self)

    def toString(self):
        s = ""
        s += "class: " + str(self.__class__.__name__) + "\n"
        s += "activation func: " + str(self.actFunc)
        # s += "preactivation: " + self.preactivation.shape + "\n"
        # s += "activations: " + self.activation + "\n"
        # s += "weights: " + self.weights + "\n"
        # s += "bias: " + self.bias + "\n"
        # s += "weights momentum: " + self.weightsMomentum + "\n"
        # s += "weights variance: " + self.weightsVariance + "\n"
        # s += "bias momentum: " + self.biasMomentum + "\n"
        # s += "bias variance: " + self.biasVariance + "\n"
        # s += "gradient wrt weights: " + self.gradientWrtWeights + "\n"
        # s += "gradient wrt bias: " + self.gradientWrtBias

        return s

    