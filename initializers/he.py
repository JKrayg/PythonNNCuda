import math
import cupy as cp
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from layers.layer import Layer


class He:
    def __init__(self):
        print("he")

    def initWeights(rows: int, cols: int):
        std = math.sqrt(2.0 / rows)
        return cp.random.randn(rows, cols) * std

    # def initWeights(prev: "Layer", curr: "Layer"):
    #     rows = prev.numNeurons
    #     cols = curr.numNeurons
    #     std = math.sqrt(2.0 / rows)

    #     return cp.random.randn(rows, cols) * std
    
    # def initWeights(inputSize: int, curr: "Layer"):
    #     rows = inputSize
    #     cols = curr.numNeurons
    #     std = math.sqrt(2.0 / rows)

    #     return cp.random.randn(rows, cols) * std
    