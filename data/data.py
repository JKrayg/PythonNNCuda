import cupy as cp
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.utils import Utils


class Data:
    def __init__(self, data: np.array, labels: np.array):
        self.data = cp.asarray(data)
        self.labelMap = {}
        count = 0
        for i in range(len(labels)):
            if labels[i] not in self.labelMap:
                self.labelMap[labels[i]] = count
                count += 1
        self.classes = list(self.labelMap.values())
        self.labels = cp.array([self.labelMap[labels[i]] for i in range(len(labels))])
        
        if (len(self.classes) > 2):
            self.labels = self.oneHot(self.labels)
        
        # print(self.labelMap)
        # print(self.classes)
        # print(self.labels)

    def oneHot(self, labels: cp.array):
        encoded: cp.ndarray = cp.ndarray((len(labels), len(self.classes)))
        for i in range(len(labels)):
            curr: cp.ndarray = cp.zeros((len(self.classes)))
            curr[labels[i]] = 1.0
            encoded[i] = curr

        return encoded
    
    def zScore(self):
        mean = cp.mean(self.data, axis=0)
        std = cp.std(self.data, axis=0)
        self.data = (self.data - mean) / std
        

    def split(self, testSize, valSize):
        rows = self.data.shape[0]
        testSetSize = rows * testSize
        valSetSize = rows * valSize
        trainSetSize = rows - (testSetSize + valSetSize)

        self.trainData = self.data[0:trainSetSize]
        self.testData = self.data[trainSetSize:(trainSetSize+testSetSize)]
        self.valData = self.data[(trainSetSize+testSetSize):rows]
        self.trainLabels = self.labels[0:trainSetSize]
        self.testLabels = self.labels[trainSetSize:(trainSetSize+testSetSize)]
        self.valLabels = self.labels[(trainSetSize+testSetSize):rows]
        

