"""
Project AI
Joost van Amersfoort - 10021248
Otto Fabius - 5619858
"""

"""A proof of concept with scikit learn implemenation of AEVB"""

#example: python trainffscikit.py

import VariationalAutoencoder
import numpy as np
import cPickle

print "Loading data"
f = open('freyfaces.pkl','rb')
data = cPickle.load(f)
f.close()


[N,dimX] = data.shape
HU_decoder = 200
HU_encoder = 200

dimZ = 5
L = 1
learning_rate = 0.02

batch_size = 20
continuous = True
n_iter = 100
verbose = True

encoder = VariationalAutoencoder.VA(HU_decoder,HU_encoder,dimZ,learning_rate,batch_size,n_iter,L,continuous,verbose)

print "Iterating"
np.random.shuffle(data)
lowerbound = encoder.fit(data)
print lowerbound
