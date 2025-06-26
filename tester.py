import cupy as cp
from layers import *
from activations import ReLU
from training.loss.catCrossEntropy import CatCrossEntropy

d1 = Dense(8, actFunc=ReLU, inputShape=4)
d2 = Dense(8, actFunc=ReLU)
d3 = Dense(8, actFunc=ReLU, lossFunc=CatCrossEntropy)
c1 = Conv2D(10, actFunc=ReLU, inputShape=[3, 28, 28], kernelShape=[3, 3], stride=1, padding="same")
c2 = Conv2D(10, actFunc=ReLU, kernelShape=[3, 3], stride=1, padding="same")
f1= Flatten()
f1.previousShape = c1.inputShape

print(d1.actFunc, c1.inputShape, f1.previousShape)

testAct = cp.arange(16).reshape(4, 4)
testGrad = cp.arange(16).reshape(4, 4)

print("activation: ", testAct)
print("gradient: ", testGrad)
l = Layer(actFunc=ReLU, inputShape=4)
print("gradient wrt weights: ", l.gradientWeights(testAct, testGrad))
print("gradient wrt bias: ", l.gradientBias(testGrad))

print(d2.toString())
