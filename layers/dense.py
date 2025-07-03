import cupy as cp;
import numpy as np

from typing import List, Optional
from layers.flatten import Flatten
from layers.layer import Layer;
from activations.activation import Activation
from training.loss.loss import Loss
from utils.utils import Utils

class Dense(Layer):
    def __init__(self, numNeurons: int, actFunc: Activation, inputShape: Optional[List[int]] = None, lossFunc: Loss = None):
        super().__init__(actFunc, inputShape)
        self.numNeurons = numNeurons
        self.lossFunc = lossFunc
        self.numFeatures = inputShape


    def initLayer(self, prev: Layer, batchSize: int):
        actFunc: Activation = self.actFunc
        self.activation = cp.empty((batchSize, self.numNeurons))
        self.preactivation = cp.empty((batchSize, self.numNeurons))

        if prev != None:
            w, b = actFunc.initWB(prev.numNeurons, self.numNeurons)
            self.weights = w
            self.bias = b
        else:
            w, b = actFunc.initWB(self.numFeatures, self.numNeurons)
            self.weights = w
            self.bias = b


        if self.normalizer != None:
            scVar = cp.fill((self.numNeurons, 1), 1)
            shMeans = cp.ndarray((self.numNeurons, 1))
            self.normalizer.scale = \
                self.normalizer.variances = \
                self.normalizer.runVar = scVar
            self.normalizer.shift = \
                self.normalizer.means = \
                self.normalizer.runMeans = shMeans
            

        return self
    

    def adamInit(self):
        w = cp.zeros((self.weights.shape[0], self.weights.shape[1]))
        b = cp.zeros((self.bias.shape[0]))


        self.weightsMomentum = self.weightsVariance = w
        self.biasMomentum = self.biasVariance = b

        norm = self.normalizer

        if (norm != None):
            shift = cp.ndarray(norm.shift[0], norm.shift[1])
            scale = cp.ndarray(norm.scale[0], norm.scale[1])

            norm.shiftMomentum = norm.shiftVariance = shift
            norm.scaleMomentum = norm.scaleVariance = scale

    
    def forwardProp(self):
        maths = Utils()
        z: cp.ndarray = maths.weightedSum(self.prev.activation, self)
        self.preactivation = z

        if (self.normalizer != None):
            z = self.normalizer.normalize(z)

        activated: cp.ndarray
        if (self.actFunc != None):
            activated = self.actFunc.activate(z)
        else:
            activated = z

        if (self.regularizers != None):
            for r in self.regularizers:
                activated = r.regularize(activated)

        self.activation = activated


    def getGradients(self, gradient: cp.ndarray, data: cp.ndarray):
        print("yes")
        gradW: cp.ndarray
        gradB: cp.ndarray
        grad: cp.ndarray

        if (self.normalizer != None):
            norm = self.normalizer
            grad = norm.gradientPreBN()
            norm.shiftGradient = norm.getShiftGrad(gradient)
            norm.scaleGradient = norm.getScaleGrad(gradient)
        else:
            grad = gradient
        
        if (self.next == None):
            gradW = self.gradientWeights(self.prev.activation, gradient)
            gradB = self.gradientBias(gradient)
        else:
            p: Layer
            if (self.prev != None):
                p = self.prev
            else:
                p = Layer()
                p.activation = data

            gradW = self.gradientWeights(self.prev.activation, gradient)
            gradB = self.gradientBias(gradient)
         

        # bad
        if (self.regularizers != None):
            for r in self.regularizers:
                gradW = gradW + r.regularize(self.weights)
                break


        self.gradientWrtWeights = gradW
        self.gradientWrtBias = gradB
        print(self.prev)

        if (self.prev != None):
            gradWrtPreAct: cp.ndarray
            g = grad @ self.weights.transpose()
            if (not isinstance(self.prev, Flatten)):
                gradWrtPreAct = self.prev.actFunc.gradient(
                    self.prev.preactivation, g)
            else:
                gradWrtPreAct = self.prev.reshapeGrad(g)
                
            
            # self.prev.getGradients(gradWrtPreAct, data)