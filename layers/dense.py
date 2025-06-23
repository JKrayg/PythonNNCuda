import cupy as cp;
import numpy as np

from typing import List, Optional
from layers.layer import Layer;
from activations.activation import Activation
from training.loss.loss import Loss

class Dense(Layer):
    def __init__(self, numNeurons: int, actFunc: Activation, inputShape: Optional[List[int]] = None, lossFunc: Loss = None):
        super().__init__(actFunc, inputShape)
        self.numNeurons = numNeurons
        self.lossFunc = lossFunc