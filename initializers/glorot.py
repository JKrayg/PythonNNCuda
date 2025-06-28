import cupy as cp
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from layers.layer import Layer

class Glorot:
    def __init__(self):
        print("glorot")

    def initWeights(rows: int, cols: int):
        return cp.random.randn(rows, cols) * math.sqrt(1 / (rows + cols))
    
    # def initWeights(inputSize: int, curr: "Layer"):
    #     rows = inputSize
    #     cols = curr.numNeurons

    #     varW = 1 / (rows + cols)

    #     return cp.random.randn(rows, cols) * math.sqrt(varW)