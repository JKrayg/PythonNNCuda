import cupy as cp

from data.data import Data
from layers.layer import Layer
from training.callbacks.earlyStopping import EarlyStopping
from training.callbacks.lrScheduler import LRScheduler
from training.optimizers.adam import Adam

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

    def shuffle(self, data: cp.ndarray, labels: cp.ndarray):
        indcs = cp.random.permutation(data.shape[0])

        return data[indcs], labels[indcs]

    def fit(self, data: Data, epochs: int, batchSize: int = 1):
        trainData: cp.ndarray
        trainLabels: cp.ndarray
        testData: cp.ndarray
        testLabels: cp.ndarray
        valData: cp.ndarray
        valLabels: cp.ndarray

        for i in range(len(self.layers)):
            prevNumNeur: int = None
            curr = self.layers[i + 1]
            if i != 0:
                prevNumNeur = self.layers[i - 1].numNeurons

            curr.initLayer(prevNumNeur, batchSize)

            if isinstance(self.optimizer, Adam):
                curr.initForAdam()


        reshape = False
        trainShape = trainData.shape

        # arrtoShufl: list[cp.ndarray]

        # if (len(trainShape) == 2):

        # erly: EarlyStopping = None
        # lrSch: LRScheduler = None

        # if (self.callbacks != None):
        #     for c in self.callbacks:
        #         if (isinstance(lrSch, LRScheduler)):
        #             lrSch = 

        for i in range(epochs):
            print("epoch", str(i + 1), "/", epochs)
            self.lossHistory = None
            trData, trLabels

            if (len(trainShape) == 2):
                trData, trLabels = self.shuffle(trainData, trainLabels)
            else:
                trData, trLabels = self.shuffle(
                    trainData.reshape(trainShape[0], trainShape[1]*trainShape[2]),
                    trainLabels)
                
            print(trData)
            print(trLabels)

            if (reshape):
                trData.reshape(trainShape)
            



