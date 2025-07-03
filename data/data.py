import cupy as cp
import numpy as np


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
            self.oneHot(self.labels)
        
        print(self.labelMap)
        print(self.classes)
        print(self.labels)

    def oneHot(self, labels: cp.array):
        encoded: cp.ndarray = cp.ndarray((len(labels), len(self.classes)))
        for i in range(len(labels)):
            curr: cp.ndarray = cp.zeros((len(self.classes)))
            print(curr[labels[i]])
            curr[labels[i]] = 1.0
            encoded[i] = curr

        print(encoded)

