##Variational Auto-encoder

This is an implementation of the paper [Stochastic Gradient VB and the Variational Auto-Encoder](http://arxiv.org/abs/1312.6114) by D. Kingma and Prof. Dr. M. Welling.


There are three different versions of the implementation. The first two versions follow the scikit-learn API, the third version is built with Theano and uses a custom API. The difference between the two scikit-learn versions is their dependence on scikit-learn itself. `VariationalAutoencoder_noSK.py` is indepedent of scikit-learn, while `VariationalAutoencoder_withSK.py` uses some private functions of scikit-learn, for batches and checking of inputs. I recommend using the `withSK` version as it's a tiny bit faster and extra checks don't hurt.

There are demos for all versions. Although it is necessary to put the data files in the right place and to import the right version of the Variational Auto-encoder. I assume this is no problem.

Please report any bugs you find as an issue on this repository or by email, see header of code. I also happily answer any question.

The code is MIT licensed.

