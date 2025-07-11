import random
import cupy as cp
import time

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
    

    def compile(self, optimizer: Optimizer = SGD,
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
            # epochStart = time.time()
            print("epoch", (i + 1), "/", epochs)
            self.lossHistory = []
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

            his = cp.array(self.lossHistory)
            self.loss = cp.sum(his) / his.shape[0]
            # self.valLoss = self.lozz(valData, valLabels)
            # acc: float = self.accuracy(trData, trLabels)
            # valAcc: float = self.accuracy(valData, valLabels)

            print("loss:", self.loss,
                #   "~ accuracy:", acc,
                #   "~ val loss:", self.valLoss,
                #   "~ val accuracy:", valAcc
                )
            print("____________________________________________________________")
            # epochEnd = time.time()
            # print(f"epoch time: {epochEnd - epochStart:.4f} seconds")

            # callbacks and lr scheduler


    def forwardPass(self, data, labels):
        # cp.cuda.Device().synchronize()
        # forwStart = time.time()
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
        # cp.cuda.Device().synchronize()
        # forwEnd = time.time()
        # print(f"forward time: {forwEnd - forwStart:.4f} seconds")
    
    def backprop(self, data, labels):
        # cp.cuda.Device().synchronize()
        # backStart = time.time()
        out: Layer = self.layers[len(self.layers) - 1]
        lossFunc: Loss = out.lossFunc
        loss: float = lossFunc.execute(out.activation, labels)
        self.lossHistory.append(loss)
        gWrtO = lossFunc.gradient(out.activation, labels)

        out.getGradients(gWrtO, data)

        for l in self.layers:
            l.updateParams(self.optimizer)

            if (isinstance(l.normalizer, BatchNormalization)):
                l.normalizer.updateShift(self.optimizer)
                l.normalizer.updateScale(self.optimizer)
        
        # cp.cuda.Device().synchronize()
        # backEnd = time.time()
        # print(f"backward time: {backEnd - backStart:.4f} seconds")


    def lozz(self, data, labels):
        # print(data)
        # print(labels)
        self.forwardPass(data, labels)
        out: Layer = self.layers[len(self.layers) - 1]
        return out.lossFunc.execute(out.activation, labels)
    
    def accuracy(self, data, labels):
        # **
        return random.uniform(0, 9)


