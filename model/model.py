from data.data import Data
from layers.layer import Layer

class Model:
    def __init__(self):
        self.layers = []
    
    def addLayer(self, l: Layer):
        numL = len(self.layers)
        if numL > 0:
            p: Layer = self.layers[numL - 1]
            p.next = l
            l.prev = p
        
        self.layers.append(l)

    # def fit(data: Data, epochs: int, batchSize: int = 1):
