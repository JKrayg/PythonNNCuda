import numpy as np
from sklearn.datasets import load_iris
from activations.sigmoid import Sigmoid
from activations.softmax import Softmax
from data.data import Data
from layers import *
from activations import ReLU
from model.model import Model
from training.loss.catCrossEntropy import CatCrossEntropy
import time

from training.optimizers.adam import Adam

model_ = Model()

d1 = Dense(32, actFunc=ReLU, inputShape=4)
d2 = Dense(64, actFunc=ReLU)
d3 = Dense(3, actFunc=Softmax, lossFunc=CatCrossEntropy)
model_.addLayer(d1)
model_.addLayer(d2)
model_.addLayer(d3)

iris = load_iris()

data = np.array(iris.data)
labels = np.array(iris.target)


# data = Data(np.random.rand(4, 4), np.array(["b", "b", "c", "d", "d", "c", "b", "b",
#                                                               "b", "b"]))



data = Data(data, labels)
data.shuffle()
data.zScore()
data.split(0.1, 0.1)

# d3.getGradients(data.data, data.labels)
model_.compile(optimizer=Adam(0.001))


model_.fit(data, 8, 20)

# for l in model_.layers:
#     print(l.toString())





# for i in model_.layers:
    
#     print("prev:", i.prev.__class__)
#     print("curr:", i.__class__)
#     print("next:", i.next.__class__)
#     print("------------------------------------------")
