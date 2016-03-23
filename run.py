import numpy as np
import time
import os
from VAE import VAE
import cPickle
import gzip

np.random.seed(42)

hu_encoder = 400
hu_decoder = 400
n_latent = 20
continuous = False
n_epochs = 40

if continuous:
    print "Loading Freyface data"
    # Retrieved from: http://deeplearning.net/data/mnist/mnist.pkl.gz
    f = open('freyfaces.pkl', 'rb')
    x = cPickle.load(f)
    f.close()
    x_train = x[:1500]
    x_valid = x[1500:]
else:
    print "Loading MNIST data"
    # Retrieved from: http://deeplearning.net/data/mnist/mnist.pkl.gz
    f = gzip.open('mnist.pkl.gz', 'rb')
    (x_train, t_train), (x_valid, t_valid), (x_test, t_test) = cPickle.load(f)
    f.close()

path = "./"

print "instantiating model"
model = VAE(continuous, hu_encoder, hu_decoder, n_latent, x_train)


batch_order = np.arange(int(model.N / model.batch_size))
epoch = 0
LB_list = []

if os.path.isfile(path + "params.pkl"):
    print "Restarting from earlier saved parameters!"
    model.load_parameters(path)
    LB_list = np.load(path + "LB_list.npy")
    epoch = len(LB_list)

if __name__ == "__main__":
    print "iterating"
    while epoch < n_epochs:
        epoch += 1
        start = time.time()
        np.random.shuffle(batch_order)
        LB = 0.

        for batch in batch_order:
            batch_LB = model.update(batch, epoch)
            LB += batch_LB

        LB /= len(batch_order)

        LB_list = np.append(LB_list, LB)
        print "Epoch {0} finished. LB: {1}, time: {2}".format(epoch, LB, time.time() - start)
        np.save(path + "LB_list.npy", LB_list)
        model.save_parameters(path)

    valid_LB = model.likelihood(x_valid)
    print "LB on validation set: {0}".format(valid_LB)



