"""
Joost van Amersfoort - <joost.van.amersfoort@gmail.com>
Otto Fabius - <ottofabius@gmail.com
"""

import numpy as np
import theano as th
import theano.tensor as T

"""This class implements an auto-encoder with Variational Bayes"""

class VA:
    def __init__(self, HU_decoder, HU_encoder, dimX, dimZ, batch_size, L=1, learning_rate=0.01):
        self.HU_decoder = HU_decoder
        self.HU_encoder = HU_encoder

        self.dimX = dimX
        self.dimZ = dimZ
        self.L = L
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.sigmaInit = 0.01
        self.lowerbound = 0

        self.continuous = False


    def initParams(self):
    	"""Initialize weights and biases, depending on if continuous data is modeled an extra weight matrix is created"""
        W1 = np.random.normal(0,self.sigmaInit,(self.HU_encoder,self.dimX))
        b1 = np.random.normal(0,self.sigmaInit,(self.HU_encoder,1))

        W2 = np.random.normal(0,self.sigmaInit,(self.dimZ,self.HU_encoder))
        b2 = np.random.normal(0,self.sigmaInit,(self.dimZ,1))

        W3 = np.random.normal(0,self.sigmaInit,(self.dimZ,self.HU_encoder))
        b3 = np.random.normal(0,self.sigmaInit,(self.dimZ,1))
        
        W4 = np.random.normal(0,self.sigmaInit,(self.HU_decoder,self.dimZ))
        b4 = np.random.normal(0,self.sigmaInit,(self.HU_decoder,1))

        W5 = np.random.normal(0,self.sigmaInit,(self.dimX,self.HU_decoder))
        b5 = np.random.normal(0,self.sigmaInit,(self.dimX,1))

        if self.continuous:
            W6 = np.random.normal(0,self.sigmaInit,(self.dimX,self.HU_decoder))
            b6 = np.random.normal(0,self.sigmaInit,(self.dimX,1))
            self.params = [W1,W2,W3,W4,W5,W6,b1,b2,b3,b4,b5,b6]
        else:
	        self.params = [W1,W2,W3,W4,W5,b1,b2,b3,b4,b5]

        self.h = [0.01] * len(self.params)


    def initH(self,miniBatch):
    	"""Compute the gradients and use this to initialize h"""
        totalGradients = self.getGradients(miniBatch)
        for i in xrange(len(totalGradients)):
            self.h[i] += totalGradients[i]*totalGradients[i]

    def createGradientFunctions(self):
        #Create the Theano variables
        W1,W2,W3,W4,W5,W6,x,eps = T.dmatrices("W1","W2","W3","W4","W5","W6","x","eps")

        #Create biases as cols so they can be broadcasted for minibatches
        b1,b2,b3,b4,b5,b6 = T.dcols("b1","b2","b3","b4","b5","b6")

        if self.continuous:
            h_encoder = T.nnet.softplus(T.dot(W1,x) + b1)
        else:   
            h_encoder = T.tanh(T.dot(W1,x) + b1)

        mu_encoder = T.dot(W2,h_encoder) + b2
        log_sigma_encoder = 0.5*(T.dot(W3,h_encoder) + b3)

        #Find the hidden variable z
        z = mu_encoder + T.exp(log_sigma_encoder)*eps

        prior = 0.5* T.sum(1 + 2*log_sigma_encoder - mu_encoder**2 - T.exp(2*log_sigma_encoder))


        #Set up decoding layer
        if self.continuous:
            h_decoder = T.nnet.softplus(T.dot(W4,z) + b4)
            mu_decoder = T.nnet.sigmoid(T.dot(W5,h_decoder) + b5)
            log_sigma_decoder = 0.5*(T.dot(W6,h_decoder) + b6)
            logpxz = T.sum(-(0.5 * np.log(2 * np.pi) + log_sigma_decoder) - 0.5 * ((x - mu_decoder) / T.exp(log_sigma_decoder))**2)
            gradvariables = [W1,W2,W3,W4,W5,W6,b1,b2,b3,b4,b5,b6]
        else:
            h_decoder = T.tanh(T.dot(W4,z) + b4)
            y = T.nnet.sigmoid(T.dot(W5,h_decoder) + b5)
            logpxz = -T.nnet.binary_crossentropy(y,x).sum()
            gradvariables = [W1,W2,W3,W4,W5,b1,b2,b3,b4,b5]


       logp = logpxz + prior

        #Compute all the gradients
        derivatives = T.grad(logp,gradvariables)

        #Add the lowerbound so we can keep track of results
        derivatives.append(logp)

        self.gradientfunction = th.function(gradvariables + [x,eps], derivatives, on_unused_input='ignore')
        self.lowerboundfunction = th.function(gradvariables + [x,eps], logp, on_unused_input='ignore')

    def iterate(self, data):
       	"""Main method, slices data in minibatches and performs an iteration"""
        [N,dimX] = data.shape
        batches = np.arange(0,N,self.batch_size)
        if batches[-1] != N:
            batches = np.append(batches,N)

        for i in xrange(0,len(batches)-2):
            miniBatch = data[batches[i]:batches[i+1]]
            totalGradients = self.getGradients(miniBatch.T)
            self.updateParams(totalGradients,N,miniBatch.shape[0])

    def getLowerBound(self,data):
    	"""Use this method for example to compute lower bound on testset"""
        lowerbound = 0
        [N,dimX] = data.shape
        batches = np.arange(0,N,self.batch_size)
        if batches[-1] != N:
            batches = np.append(batches,N)

        for i in xrange(0,len(batches)-2):
            miniBatch = data[batches[i]:batches[i+1]]
            e = np.random.normal(0,1,[self.dimZ,miniBatch.shape[0]])
            lowerbound += self.lowerboundfunction(*(self.params),x=miniBatch.T,eps=e)

        return lowerbound/N


    def getGradients(self,miniBatch):
    	"""Compute the gradients for one minibatch and check if these do not contain NaNs"""
        totalGradients = [0] * len(self.params)
        for l in xrange(self.L):
            e = np.random.normal(0,1,[self.dimZ,miniBatch.shape[1]])
            gradients = self.gradientfunction(*(self.params),x=miniBatch,eps=e)
            self.lowerbound += gradients[-1]

            for i in xrange(len(self.params)):
                totalGradients[i] += gradients[i]

        return totalGradients

    def updateParams(self,totalGradients,N,current_batch_size):
    	"""Update the parameters, taking into account AdaGrad and a prior"""
        for i in xrange(len(self.params)):
            self.h[i] += totalGradients[i]*totalGradients[i]
            if i < 5 or (i < 6 and len(self.params) == 12):
                prior = 0.5*self.params[i]
            else:
                prior = 0

            self.params[i] += self.learning_rate/np.sqrt(self.h[i]) * (totalGradients[i] - prior*(current_batch_size/N))
