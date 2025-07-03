from typing import List, Optional
import cupy as cp;

from activations.activation import Activation
from training.loss.loss import Loss
from training.optimizers.optimizer import Optimizer

class Layer:
    def __init__(self, actFunc: Optional[Activation] = None, inputShape: Optional[List[int]] = None):
        self.next = None
        self.prev = None
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
        self.normalizer = None

    def gradientWeights(self, activation: cp.ndarray, gradient: cp.ndarray):
        gWrTw = (cp.transpose(activation) @ gradient) / activation.shape[0]
        return gWrTw
    
    def gradientBias(self, gradient: cp.ndarray):
        sum_ = gradient.sum(axis=0).reshape(gradient.shape[1], 1) / gradient.shape[0]
        return sum_
    
    def updateBias(self, o: Optimizer):
        self.bias = o.biasUpdate(self)

    def toString(self):
        s = ""
        s += "class: " + str(self.__class__.__name__) + "\n"
        s += "activation func: " + str(self.actFunc.__qualname__) + "\n"
        s += "preactivation: " + str(self.preactivation.shape) + "\n"
        s += "activations: " + str(self.activation.shape) + "\n"
        s += "weights: " + str(self.weights.shape) + "\n"
        s += "bias: " + str(self.bias.shape) + "\n"
        s += "weights momentum: " + str(self.weightsMomentum.shape) + "\n"
        s += "weights variance: " + str(self.weightsVariance.shape) + "\n"
        s += "bias momentum: " + str(self.biasMomentum.shape) + "\n"
        s += "bias variance: " + str(self.biasVariance.shape) + "\n"
        # s += "gradient wrt weights: " + self.gradientWrtWeights + "\n"
        # s += "gradient wrt bias: " + self.gradientWrtBias

        return s

    