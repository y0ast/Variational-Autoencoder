"""
Project AI
Joost van Amersfoort - 10021248
Otto Fabius - 5619858
"""

"""Demonstration comparing SGVB to RBM for feature extraction and classification"""

import sgvb
import numpy as np
import gzip,cPickle
from sklearn import neural_network, linear_model

load_params = False

print "Loading data"
#Retrieved from: http://deeplearning.net/data/mnist/mnist.pkl.gz

f = gzip.open('mnist.pkl.gz', 'rb')
(x_train, t_train), (x_valid, t_valid), (x_test, t_test)  = cPickle.load(f)
f.close()

data_train, data_test = x_valid[:1000], x_test[:1000]
t_train, t_test = t_valid[:1000], t_test[:1000]

if load_params == True:
	print 'Loading learned weights with SGVB and RBM from file, calculating features'
	#load params, transform data_train, data_test

else:
	print 'extracting features with SGVB auto-encoder, default is 10 iterations...'
	encoder = sgvb.SGVB(verbose=True)
	encoder.fit(x_train)
	SGVB_train_features = encoder.transform(data_train)
	SGVB_test_features  = encoder.transform(data_test)
	print 'done'
	
	print 'extracting features with RBM...'
	n_components=200
	learning_rate = 0.01
	batch_size=100
	rbm = neural_network.BernoulliRBM(n_components,learning_rate,batch_size, verbose=True)
	rbm.fit(x_train)
	RBM_train_features = rbm.transform(data_train)
	RBM_test_features  = rbm.transform(data_test)
	print 'done'

print 'performing logistic regression on raw data...'
LogReg_raw = linear_model.LogisticRegression()
LogReg_raw.fit(data_train,t_train)
raw_score = LogReg_raw.score(data_test, t_test)
print 'Test score on raw data = ', raw_score

print 'performing logistic regression on RBM features...'
LogReg_RBM = linear_model.LogisticRegression()
LogReg_RBM.fit(RBM_train_features,t_train)
RBM_score = LogReg_RBM.score(RBM_test_features, t_test)
print 'Test score on RBM features = ', RBM_score

print 'performing logistic regression on SGVB features...'
LogReg_SGVB = linear_model.LogisticRegression()
LogReg_SGVB.fit(SGVB_train_features,t_train)
SGVB_score = LogReg_SGVB.score(SGVB_test_features, t_test)
print 'Test score on SGVB features = ', SGVB_score
