import numpy as np
import theano
import theano.tensor as T

import cPickle
from collections import OrderedDict


class VAE:
    """This class implements the Variational Auto Encoder"""
    def __init__(self, continuous, hu_encoder, hu_decoder, latent_variables, x_train, b1=0.95, b2=0.999, batch_size=100, learning_rate=0.01, lam=0):
        self.continuous = continuous
        self.hu_encoder = hu_encoder
        self.hu_decoder = hu_decoder
        self.latent_variables = latent_variables
        [self.N, self.features] = x_train.shape

        self.b1 = theano.shared(np.array(b1).astype(theano.config.floatX), name='b1')
        self.b2 = theano.shared(np.array(b2).astype(theano.config.floatX), name='b2')
        self.learning_rate = theano.shared(np.array(learning_rate).astype(theano.config.floatX), name="learning_rate")

        self.batch_size = batch_size

        sigma_init = 0.05

        create_weight = lambda dim_input, dim_output: np.random.normal(0, sigma_init, (dim_input, dim_output)).astype(theano.config.floatX)
        create_bias = lambda dim_output: np.zeros(dim_output).astype(theano.config.floatX)

        # encoder
        W_xh = theano.shared(create_weight(self.features, hu_encoder), name='W_xh')
        b_xh = theano.shared(create_bias(hu_encoder), name='b_xh')

        W_hmu = theano.shared(create_weight(hu_encoder, latent_variables), name='W_hmu')
        b_hmu = theano.shared(create_bias(latent_variables), name='b_hmu')

        W_hsigma = theano.shared(create_weight(hu_encoder, latent_variables), name='W_hsigma')
        b_hsigma = theano.shared(create_bias(latent_variables), name='b_hsigma')

        # decoder
        W_zh = theano.shared(create_weight(latent_variables, hu_decoder), name='W_zh')
        b_zh = theano.shared(create_bias(hu_decoder), name='b_zh')

        self.params = OrderedDict([("W_xh", W_xh), ("b_xh", b_xh), ("W_hmu", W_hmu), ("b_hmu", b_hmu),
                                   ("W_hsigma", W_hsigma), ("b_hsigma", b_hsigma), ("W_zh", W_zh),
                                   ("b_zh", b_zh)])

        if self.continuous:
            W_hxmu = theano.shared(create_weight(hu_decoder, self.features), name='W_hxmu')
            b_hxmu = theano.shared(create_bias(self.features), name='b_hxmu')

            W_hxsig = theano.shared(create_weight(hu_decoder, self.features), name='W_hxsig')
            b_hxsig = theano.shared(create_bias(self.features), name='b_hxsig')

            self.params.update({'W_hxmu': W_hxmu, 'b_hxmu': b_hxmu, 'W_hxsig': W_hxsig, 'b_hx': b_hxsig})
        else:
            W_hx = theano.shared(create_weight(hu_decoder, self.features), name='W_hx')
            b_hx = theano.shared(create_bias(self.features), name='b_hx')

            self.params.update({'W_hx': W_hx, 'b_hx': b_hx})

        # Adam parameters
        self.m = OrderedDict()
        self.v = OrderedDict()

        for key, value in self.params.items():
                self.m[key] = theano.shared(np.zeros_like(value.get_value()).astype(theano.config.floatX), name='m_' + key)
                self.v[key] = theano.shared(np.zeros_like(value.get_value()).astype(theano.config.floatX), name='v_' + key)

        print "Compiling Theano function!"
        x_train = theano.shared(x_train.astype(theano.config.floatX), name="x_train")

        self.update, self.encode, self.decode = self.create_gradientfunctions(x_train)

    def create_gradientfunctions(self, x_train):
        """This function takes as input the whole dataset and creates the entire model"""
        x = T.matrix("x")

        h_encoder = T.nnet.softplus(T.dot(x, self.params['W_xh']) + self.params['b_xh'].dimshuffle('x', 0))

        mu = T.dot(h_encoder, self.params['W_hmu']) + self.params['b_hmu'].dimshuffle('x', 0)
        log_sigma = T.dot(h_encoder, self.params['W_hsigma']) + self.params['b_hsigma'].dimshuffle('x', 0)

        seed = 42

        if "gpu" in theano.config.device:
            srng = theano.sandbox.cuda.rng_curand.CURAND_RandomStreams(seed=seed)
        else:
            srng = T.shared_randomstreams.RandomStreams(seed=seed)

        eps = srng.normal(mu.shape)

        # Reparametrize
        z = mu + T.exp(0.5 * log_sigma) * eps

        # Set up decoding layer
        h_decoder = T.nnet.softplus(T.dot(z, self.params['W_zh']) + self.params['b_zh'].dimshuffle('x', 0))

        if self.continuous:
            mu_decoder = T.nnet.sigmoid(T.dot(h_decoder, self.params['W_hxmu']) + self.params['b_hxmu'].dimshuffle('x', 0))
            log_sigma_decoder = T.dot(h_decoder, self.params['W_hxsig']) + self.params['b_hxsig']

            logpxz = (-(0.5 * np.log(2 * np.pi) + 0.5 * log_sigma_decoder) -
                      0.5 * ((x - mu_decoder)**2 / T.exp(log_sigma_decoder))).sum(axis=1, keepdims=True)
        else:
            reconstructed_x = T.nnet.sigmoid(T.dot(h_decoder, self.params['W_hx']) + self.params['b_hx'].dimshuffle('x', 0))
            logpxz = - T.nnet.binary_crossentropy(reconstructed_x, x).sum(axis=1, keepdims=True)

        # Entropy of log q(z|x,y)
        logqz_xy = - 0.5 * T.sum(np.log(2 * np.pi) + 1 + log_sigma, axis=1, keepdims=True)

        # integral over q(z|x,y) log p(z)
        logpz = - 0.5 * T.sum(np.log(2 * np.pi) + (mu**2 + T.exp(log_sigma)), axis=1, keepdims=True)

        # Legacy code from the normal VAE: Expectation of (logpz - logqz_xy) over logqz_xy is equal to the KLD:
        # KLD = 0.5 * T.sum(1 + log_sigma - mu**2 - T.exp(log_sigma))

        # Average over batch dimension
        logpx = T.mean(logpxz + logpz - logqz_xy)

        # Compute all the gradients
        gradients = T.grad(logpx, self.params.values())

        # Adam implemented as updates
        updates = OrderedDict()
        epoch = T.iscalar("epoch")
        gamma = T.sqrt(1 - (1 - self.b2)**epoch)/(1 - (1 - self.b1)**epoch)

        # Adam
        for parameter, gradient, m, v in zip(self.params.values(), gradients, self.m.values(), self.v.values()):
            new_m = self.b1 * gradient + (1 - self.b1) * m
            new_v = self.b2 * (gradient**2) + (1 - self.b2) * v

            updates[parameter] = parameter + self.learning_rate * gamma * new_m / (T.sqrt(new_v) + 1e-8)
            updates[m] = new_m
            updates[v] = new_v

        batch = T.iscalar('batch')

        givens = {
            x: x_train[batch*self.batch_size:(batch+1)*self.batch_size, :]
        }

        update = theano.function([batch, epoch], logpx, updates=updates, givens=givens)
        encode = theano.function([x], z)
        decode = theano.function([z], reconstructed_x)

        return update, encode, decode

    def transform_data(self, x_train):
        transformed_x = np.zeros((self.N, self.latent_variables))
        batches = np.arange(int(self.N / self.batch_size))

        for batch in batches:
            batch_x = x_train[batch*self.batch_size:(batch+1)*self.batch_size, :]
            transformed_x[batch*self.batch_size:(batch+1)*self.batch_size, :] = self.encode(batch_x)

        return transformed_x


    def save_parameters(self, path):
        """Saves all the parameters in a way they can be retrieved later"""
        cPickle.dump({name: p.get_value() for name, p in self.params.items()}, open(path + "/params.pkl", "wb"))
        cPickle.dump({name: m.get_value() for name, m in self.m.items()}, open(path + "/m.pkl", "wb"))
        cPickle.dump({name: v.get_value() for name, v in self.v.items()}, open(path + "/v.pkl", "wb"))

    def load_parameters(self, path):
        """Load the variables in a shared variable safe way"""
        p_list = cPickle.load(open(path + "/params.pkl", "rb"))
        m_list = cPickle.load(open(path + "/m.pkl", "rb"))
        v_list = cPickle.load(open(path + "/v.pkl", "rb"))

        for name in p_list.keys():
            self.params[name].set_value(p_list[name].astype(theano.config.floatX))
            self.m[name].set_value(m_list[name].astype(theano.config.floatX))
            self.v[name].set_value(v_list[name].astype(theano.config.floatX))
