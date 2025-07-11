import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_openml
from activations.sigmoid import Sigmoid
from activations.softmax import Softmax
from data.data import Data
from layers import *
from activations import ReLU
from model.model import Model
from training.loss.catCrossEntropy import CatCrossEntropy
import time

from training.optimizers.adam import Adam

iris = load_iris()
# x, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
x = np.array(iris.data)

y = np.array(iris.target)
data = Data(x, y)
data.shuffle()
data.zScore()
data.split(0.1, 0.1)

model_ = Model()
d1 = Dense(16, actFunc=ReLU, inputShape=4)
d2 = Dense(8, actFunc=ReLU)
# d4 = Dense(32, actFunc=ReLU)
d3 = Dense(3, actFunc=Softmax, lossFunc=CatCrossEntropy)
model_.addLayer(d1)
model_.addLayer(d2)
# model_.addLayer(d4)
model_.addLayer(d3)

model_.compile(optimizer=Adam(0.001))
model_.fit(data, 8, 20)
