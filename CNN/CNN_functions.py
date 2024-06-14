#This is all of the algorthims used in the CNN layers but done manually without the help of a ML lib

# TODO: research back propagation as well as additional functions that may need to be added to this

import numpy as np
from utils import crossCorrelation


class Layer:

    def __init__(self, dimensionIn, dimensionOut, lr):
        
        self.dimensionOut = dimensionOut
        self.dimensionIn = dimensionIn
        self.lr = lr

    def forward(self, inputs):
        raise NotImplementedError
    
    def backward(self, gradients):
        raise NotImplementedError

    def initialWeights(self):
        pass
    
    def weights(self):
        pass
'''
    
    uses cross correlation function written in Utils to calculate cross-correlation for each filter

'''
class convolution(Layer):

    def __init__(self, dimensionIn, dimensionFilter, numFilters, lr=0.001):

        self.dimensionIn = dimensionIn
        self.dimensionFilter = dimensionFilter
        self.numFilters = numFilters
        self.lr = lr

    def forward(self, a):

        self.a = a
        b = crossCorrelation(a, self.filters)

        return b
    
    def backward(self, Z):

        Zx = crossCorrelation(Z, self.filters)[1]
        Zf = crossCorrelation(self.a, Z)[0]
        self.weights(Zf)

        return Zx

    def initialWeights(self):
        self.filters = np.random.randn(self.filterDimension, self.filterDimension, self.numFilters)
    
    def weights(self, Z):
        self.filters -= self.lr * Z

'''
aggregates vectors through dotproduct
'''
class FullyConnectedLayer(Layer):

    def __init__(self, outputDimension, inputDimension, lr):
        super().__init__(
        outputDimension=outputDimension, 
        inputDimension=inputDimension, 
        lr=lr)

    def forward(self, a):
        
        self.a = a
        self.b = np.dot(self.y, self.a) + self.z
        return self.b

    def backward(self, Z):

        self.weights(Z)
        Zx = np.dot(self.y.T, Z)
        return Zx
    
    def initialWeights(self):

        self.y = np.random.rand(self.outputDimension, self.inputDimension)
        self.z = np.random.rand(self.outputDimension, 1)
    
    def weights(self, Z):
        self.y -= self.lr * np.dot(Z, self.a)
        self.z -= self.lr * Z


'''
ReLU activation function used in convolutional layer. 0 if x <= 0 or a if x > 0;
'''
class ReLU(Layer):

    def __init__(self):
        pass

    def forward(self, a):
        self.a = a
        b = (self.a > 0) * a

        return b
    
    def backward(self, Z):
        y = (self.a > 0) * Z
        return y
'''

    uses softmax formula and represents a probability distribution over a discrete variable with n possible values

'''
class SoftMax(Layer):
   
    def __init__(self):
        pass
   
    def forward(self, a):
        self.b = np.exp(a - np.max(a)) / np.sum(np.exp(a - np.max(a)))
        return self.b

    def backward(self, Z):
        return self.b - Z
    

'''
Fixes covariate shit problem through normalizing testing and training variable distributions
'''
class BatchNormalization(Layer):

    def __init__(self):
        pass

    def forward(self, a, b, c):
        #eps: Constant for numeric stability - Look into what it means more

        eps=1e-5
        sampleMean = a.mean(axis=0)
        sampleVar = a.var(axis=0)
        std = np.sqrt(sampleVar + eps)

        x_centered = a - sampleMean
        x_norm = x_centered / std

        result = b * x_norm + c
        
        return result
    
    # add forward backward implementation
    

'''
Calculates mean square error of the data to calcuate loss. (This is not used we use categorical cross entropy for multiclass classificaition)
'''
class MeanSquareError(Layer):

    def __init__(self):
        pass

    def forward(self, a, b):
        self.a = a
        self.b = b
        loss = np.mean(np.power(self.a - self.b, 2))

        return loss

    def backward(self):
        return -2 * (self.a - self.b)