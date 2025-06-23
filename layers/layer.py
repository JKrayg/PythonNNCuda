from typing import List, Optional
import cupy as cp;

from activations.activation import Activation
from training.loss.loss import Loss

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

    