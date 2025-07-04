import cupy as cp
import numpy as np
from activations.sigmoid import Sigmoid
from data.data import Data
from layers import *
from activations import ReLU
from model.model import Model
from training.loss.catCrossEntropy import CatCrossEntropy
import time

model_ = Model()

d1 = Dense(8, actFunc=ReLU, inputShape=4)
d2 = Dense(8, actFunc=ReLU)
d3 = Dense(8, actFunc=Sigmoid, lossFunc=CatCrossEntropy)
c1 = Conv2D(10, actFunc=ReLU, inputShape=[3, 28, 28], kernelShape=[3, 3], stride=1, padding="same")
c2 = Conv2D(10, actFunc=ReLU, kernelShape=[3, 3], stride=1, padding="same")
f1 = Flatten()

# model_.addLayer(c1)
# model_.addLayer(c2)
# model_.addLayer(f1)
model_.addLayer(d1)
model_.addLayer(d2)
model_.addLayer(d3)

d1.initLayer(None, 32)
d1.adamInit()
d2.initLayer(d1, 32)
d2.adamInit()
d3.initLayer(d2, 32)
d3.adamInit()



data = Data(np.zeros((32, 8)), np.array(["z", "fi", "t", "fi", "t", "fi", "h", "ft"]))
data.zScore()
d3.getGradients(data.data, data.labels)

model_.fit(data, 1, 1)

# for l in model_.layers:
#     print(l.toString())





# for i in model_.layers:
    
#     print("prev:", i.prev.__class__)
#     print("curr:", i.__class__)
#     print("next:", i.next.__class__)
#     print("------------------------------------------")
