import math
import cupy as cp
from training.optimizers.optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, learningRate: float, momentumDecay: float = 0.9,
                 varianceDecay: float = 0.999, epsilon: float = 1e-8):
        self.learningRate = learningRate
        self.momentumDecay = momentumDecay
        self.varianceDecay = varianceDecay
        self.epsilon = epsilon
        self.updateCount = 1

    def update(self, l):
        # print("weights ", l.weights)
        mBiasCor = 1 - math.pow(self.momentumDecay, self.updateCount)
        vBiasCor = 1 - math.pow(self.varianceDecay, self.updateCount)

        l.weightsMomentum = (l.weightsMomentum * self.momentumDecay) \
            + (l.gradientWrtWeights * (1 - self.momentumDecay))
        
        l.weightsVariance = (l.weightsVariance * self.varianceDecay) \
            + (l.gradientWrtWeights * l.gradientWrtWeights) * (1 - self.varianceDecay)
        
        w = l.weights - ( (l.weightsMomentum / mBiasCor) * self.learningRate ) / ( cp.sqrt(l.weightsVariance / vBiasCor) + self.epsilon )
        
        l.biasMomentum = (l.biasMomentum * self.momentumDecay) \
            + (l.gradientWrtBias * (1 - self.momentumDecay))
        
        l.biasVariance = (l.biasVariance * self.varianceDecay) \
            + (l.gradientWrtBias * l.gradientWrtBias) * (1 - self.varianceDecay)
        
        b = (((l.bias - l.biasMomentum) / mBiasCor) / cp.power(l.biasVariance / vBiasCor, 0.5) 
                + self.epsilon) * self.learningRate
        
        return w, b
    
    # def biasUpdate(self, l):
    #     mBiasCor = 1 - math.pow(self.momentumDecay, self.updateCount)
    #     vBiasCor = 1 - math.pow(self.varianceDecay, self.updateCount)

    #     l.biasMomentum = (l.biasMomentum * self.momentumDecay) \
    #         + (l.gradientWrtBias * (1 - self.momentumDecay))
        
    #     l.biasVariance = (l.biasVariance * self.varianceDecay) \
    #         + (l.gradientWrtBias * l.gradientWrtBias) * (1 - self.varianceDecay)
        
    #     return (((l.bias - l.biasMomentum) / mBiasCor) / cp.power(l.biasVariance / vBiasCor, 0.5) 
    #             + self.epsilon) * self.learningRate