"""
Project AI
Joost van Amersfoort - 10021248
Otto Fabius - 5619858
"""

#example: python trainffscikit.py

import VariationalAutoencoder
import numpy as np
import gzip,cPickle

print "Loading data"
f = gzip.open('mnist.pkl.gz', 'rb')
(x_train, t_train), (x_valid, t_valid), (x_test, t_test)  = cPickle.load(f)
f.close()

data = x_train

[N,dimX] = data.shape
HU_decoder = 400
HU_encoder = 400

dimZ = 20
L = 1
learning_rate = 0.01

batch_size = 100
n_iter = 100
continuous = False

verbose = True

encoder = VariationalAutoencoder.VA(HU_decoder,HU_encoder,dimZ,learning_rate,batch_size,n_iter,L,continuous,verbose)


print "Iterating"
# np.random.shuffle(data)
encoder.fit(data)
