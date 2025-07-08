import numpy as np
from sklearn.datasets import load_iris
from activations.sigmoid import Sigmoid
from data.data import Data
from layers import *
from activations import ReLU
from model.model import Model
from training.loss.catCrossEntropy import CatCrossEntropy
import time

from training.optimizers.adam import Adam

model_ = Model()

d1 = Dense(6, actFunc=ReLU, inputShape=4)
d2 = Dense(16, actFunc=ReLU)
d3 = Dense(3, actFunc=Sigmoid, lossFunc=CatCrossEntropy)
# c1 = Conv2D(10, actFunc=ReLU, inputShape=[3, 28, 28], kernelShape=[3, 3], stride=1, padding="same")
# c2 = Conv2D(10, actFunc=ReLU, kernelShape=[3, 3], stride=1, padding="same")
# f1 = Flatten()

# model_.addLayer(c1)
# model_.addLayer(c2)
# model_.addLayer(f1)
model_.addLayer(d1)
model_.addLayer(d2)
model_.addLayer(d3)

# d1.initLayer(None, 16)
# d1.adamInit()
# d2.initLayer(d1.numNeurons, 16)
# d2.adamInit()
# d3.initLayer(d2.numNeurons, 16)

# d3.adamInit()

iris = load_iris()

data = np.array(iris.data)
labels = np.array(iris.target)


# data = Data(np.random.randint(0, 10, size=(100, 4)), np.array(["b", "b", "c", "d", "d", "c", "b", "b",
                                                            #   "b", "b"]))

data = Data(data, labels)
data.split(0.1, 0.1)
# data.zScore()
# d3.getGradients(data.data, data.labels)
model_.compile(optimizer=Adam(0.001))
model_.fit(data, 10, 16)

# for l in model_.layers:
#     print(l.toString())





# for i in model_.layers:
    
#     print("prev:", i.prev.__class__)
#     print("curr:", i.__class__)
#     print("next:", i.next.__class__)
#     print("------------------------------------------")
