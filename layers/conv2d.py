from typing import List, Literal, Optional
import cupy as cp

from activations.activation import Activation
from layers.layer import Layer
from training.loss.loss import Loss

class Conv2D(Layer):
    def __init__(self, numFilters: int, kernelShape: Optional[List[int]],
                 stride: int, actFunc: Activation,
                 padding: Literal["valid", "same"] = "valid",
                 inputShape: Optional[List[int]] = None):
        
        super().__init__(actFunc, inputShape)
        self.numFilters = numFilters
        self.kernelShape = kernelShape
        self.stride = stride
        self.padding = padding
        self.filters = None
        self.gradientWrtFilters = None