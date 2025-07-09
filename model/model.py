import random
import cupy as cp

from data.data import Data
from layers.layer import Layer
from training.callbacks.callback import Callback
from training.callbacks.earlyStopping import EarlyStopping
from training.callbacks.lrScheduler import LRScheduler
from training.loss.loss import Loss
from training.metrics.metric import Metric
from training.normalization.batchNorm import BatchNormalization
from training.optimizers.adam import Adam
from training.optimizers.optimizer import Optimizer
from training.optimizers.sgd import SGD

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
    

    def compile(self, optimizer: Optimizer = SGD(),
                callbacks: list[Callback] = None, metrics: Metric = None):
        self.optimizer = optimizer
        self.callbacks = callbacks
        self.metrics = metrics


    def fit(self, data: Data, batchSize: int, epochs: int):
        out: Layer = self.layers[len(self.layers) - 1]
        trainData = data.trainData
        trainLabels = data.trainLabels
        # testData = data.testData
        # testLabels = data.testLabels
        valData = data.valData
        valLabels = data.valLabels

        for i in range(len(self.layers)):
            prevNumNeurs: int
            curr = self.layers[i]
            if i != 0:
                prevNumNeurs = curr.prev.numNeurons
            else:
                prevNumNeurs = curr.numFeatures

            curr.initLayer(prevNumNeurs, batchSize)

            if isinstance(self.optimizer, Adam):
                curr.adamInit()


        reshape = False
        trainShape = trainData.shape

        for i in range(epochs):
            print("epoch", (i + 1), "/", epochs)
            self.lossHistory = cp.empty(0)
            trData = trLabels = None

            if (len(trainShape) == 2):
                trData, trLabels = self.shuffle(trainData, trainLabels)
            else:
                reshape = True
                trData, trLabels = self.shuffle(
                    trainData.reshape(trainShape[0], trainShape[1]*trainShape[2]),
                    trainLabels)

            if (reshape):
                trData.reshape(trainShape)
            
            rows = trainShape[0]

            dataBatch: cp.ndarray = None
            labelBatch: cp.ndarray = None

            for k in range(rows - (rows % batchSize)):
                dataBatch = trData[k:(k + batchSize)]
                labelBatch = trLabels[k:(k + batchSize)]
                out.labels = labelBatch
                self.forwardPass(dataBatch, labelBatch)
                self.backprop(dataBatch, labelBatch)

                if isinstance(self.optimizer, Adam):
                    self.optimizer.updateCount += 1
            
            # last batch - bad
            if(rows % batchSize != 0):
                dataBatch = trData[batchSize - (rows % batchSize): batchSize]
                labelBatch = trLabels[batchSize - (rows % batchSize): batchSize]
                out.labels = labelBatch
                self.forwardPass(dataBatch, labelBatch)
                self.backprop(dataBatch, labelBatch)

                if isinstance(self.optimizer, Adam):
                    self.optimizer.updateCount += 1

            self.loss = cp.sum(self.lossHistory) / self.lossHistory.shape[0]
            self.valLoss = self.lozz(valData, valLabels)
            acc: float = self.accuracy(trData, trLabels)
            valAcc: float = self.accuracy(valData, valLabels)

            print("loss:", self.loss,
                #   "~ accuracy:", acc,
                  "~ val loss:", self.valLoss,
                #   "~ val accuracy:", valAcc
                )
            print("____________________________________________________________")

            # callbacks and lr scheduler


    def forwardPass(self, data, labels):
        dummy: Layer = Layer()
        dummy.activation = data
        self.layers[0].prev = dummy
        self.layers[0].forwardProp(dummy)
        for i in range(1, len(self.layers)):
            curr = self.layers[i]
            prev = self.layers[i - 1]

            curr.forwardProp(prev)

            # if (curr.next is None):
            #     curr.labels = labels
    
    def backprop(self, data, labels):
        out: Layer = self.layers[len(self.layers) - 1]
        lossFunc: Loss = out.lossFunc
        loss: float = lossFunc.execute(out.activation, labels)
        self.lossHistory = cp.append(self.lossHistory, loss)
        gWrtO = lossFunc.gradient(out.activation, labels)

        out.getGradients(gWrtO, data)

        for l in self.layers:
            l.updateWeights(self.optimizer)
            l.updateBias(self.optimizer)

            if (isinstance(l.normalizer, BatchNormalization)):
                l.normalizer.updateShift(self.optimizer)
                l.normalizer.updateScale(self.optimizer)


    def lozz(self, data, labels):
        # print(data)
        # print(labels)
        self.forwardPass(data, labels)
        out: Layer = self.layers[len(self.layers) - 1]
        return out.lossFunc.execute(out.activation, labels)
    
    def accuracy(self, data, labels):
        # **
        return random.uniform(0, 9)


