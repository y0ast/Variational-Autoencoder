##Variational Auto-encoder

This is an improved implementation of the paper [Stochastic Gradient VB and the Variational Auto-Encoder](http://arxiv.org/abs/1312.6114) by D. Kingma and Prof. Dr. M. Welling. This code uses ReLUs and the adam optimizer, instead of sigmoids and adagrad. These changes make the network converge much faster.

In my other [repository](https://github.com/y0ast/VAE-Torch) the implementation is in Torch7 (lua), this version is based on Theano (Python).
To run the MNIST experiment:

`python run.py`

Setting the continuous boolean to true will make the script run the freyfaces experiment. It is necessary to tweak the batch_size and learning rate parameter for this to run smoothly.

There used to be a scikit-learn implementation too, but it was very slow and outdated. You can still find it by looking at the code at [this commit](https://github.com/y0ast/Variational-Autoencoder/tree/5c06a7f14de7f872d837cd4268ee2d081a90056d)


The code is MIT licensed.









