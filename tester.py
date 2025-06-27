import cupy as cp
from layers import *
from activations import ReLU
from model.model import Model
from training.loss.catCrossEntropy import CatCrossEntropy

model_ = Model()

d1 = Dense(8, actFunc=ReLU, inputShape=4)
d2 = Dense(8, actFunc=ReLU)
d3 = Dense(8, actFunc=ReLU, lossFunc=CatCrossEntropy)
c1 = Conv2D(10, actFunc=ReLU, inputShape=[3, 28, 28], kernelShape=[3, 3], stride=1, padding="same")
c2 = Conv2D(10, actFunc=ReLU, kernelShape=[3, 3], stride=1, padding="same")
f1 = Flatten()
f1.previousShape = c1.inputShape

model_.addLayer(c1)
model_.addLayer(c2)
model_.addLayer(f1)
model_.addLayer(d1)
model_.addLayer(d2)

print(model_.layers)

for i in model_.layers:
    print("prev:", i.prev.__class__)
    print("curr:", i.__class__)
    print("next:", i.next.__class__)
    print("------------------------------------------")
