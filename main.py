import numpy as np
import pandas as pd
import theano
import theano.tensor as T

from theano import shared
from dataWrangling import *
from optimizers import *

class RandomUtilityModel(object):
    """ Discrete choice model class structure"""
    def __init__(self, n_in, n_out, av, genericInput=None, nongenericInput=None):
        # initialize bias c of shape (i,) (Alternative specific constants)
        self.c = shared(value=np.zeros((n_out,), dtype=theano.config.floatX), name='c', borrow=True)

        # add to utility
        self.utility = self.c
        self.params = [self.c]

        # construct parameter values for generic and non-generic inputs
        # non-generic paramters for generic variables
        if genericInput is not None:
            self.W_ng = shared(value=np.zeros((6 * n_out), dtype=theano.config.floatX), name='W_ng', borrow=True)
            genericUtility = self.linear(genericInput, self.W_ng.reshape((6, n_out)))
            self.utility += genericUtility
            self.params.extend([self.W_ng])
        if nongenericInput is not None:
            self.W_g = shared(value=np.zeros((n_in,), dtype=theano.config.floatX), name='W_g', borrow=True)
            nongenericUtility = self.linear(nongenericInput, self.W_g)
            self.utility += nongenericUtility
            self.params.extend([self.W_g])

        # estimate a Logit model
        self.p_y_given_x = self.softmax(self.utility, av)

        # prediction by max-prob
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)


        # keep track of model inputs
        self.genericInput = genericInput
        self.nongenericInput = nongenericInput

        # keep track of model parameters

    def linear(self, input, params):
        # performs a dot (inner) product of input and parameters
        return T.dot(input, params)

    def softmax(self, x, av):
        e_x = av*T.exp(x - x.max(axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True) + 1e-8

    def negLogLikelihood(self, y):
        # define the loss function
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def loglikelihood(self, y):
        # full loglikelihood
        return T.sum(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        # returns the number of errors as a percentage of total number of examples
        return T.mean(T.neq(self.y_pred, y))

    def hessian(self, y):
        # returns the hessian of the negative loglikelihood wrt parameters
        return T.hessian(cost=self.negLogLikelihood(y), wrt=self.params)

def main():
    # import dataset
    data_x_ng, data_x_g, data_y, availability = dataWrangling()
    dataset_x_ng = shared(np.asarray(data_x_ng, dtype=theano.config.floatX), borrow=True)
    dataset_x_g = shared(np.asarray(data_x_g, dtype=theano.config.floatX), borrow=True)
    dataset_y = T.cast(shared(np.asarray(data_y-1, dtype=theano.config.floatX), borrow=True), 'int32')
    dataset_av = shared(np.asarray(availability, dtype=theano.config.floatX), borrow=True)

    # check size of array
    n, i, m = data_x_ng.shape # (rows, alts, variables)

    # hyperparameters
    batch_size = n
    n_batches = 1
    n_epochs = 5000
    lr = 5e-2

    # generate symbolic variables for inputs and output
    x_ng = T.tensor3('x_ng')
    x_g = T.matrix('x_g')
    y = T.ivector('y')
    av = T.matrix('av')

    # allocate symobolic variable to index
    index = T.lscalar()

    # construct symbolic representation of the Logit function
    rum = RandomUtilityModel(n_in=m, n_out=i, av=av, genericInput=x_g, nongenericInput=x_ng)
    cost = rum.negLogLikelihood(y)

    # obtaining the gradients wrt to the loss function
    grads = [T.grad(cost=cost, wrt=subset) for subset in rum.params]

    optimizer = sgd(rum.params)
    updates = optimizer.updates(params=rum.params, grads=grads, learning_rate=lr)
    #updates = optimizer.updates(params=rum.params, grads=grads, learning_rate=lr, momentum=0.9)

    # compile the theano function
    estimate_model = theano.function(
        inputs=[index],
        outputs=rum.errors(y),
        updates=updates,
        on_unused_input='ignore',
        givens={
            x_ng: dataset_x_ng[index*batch_size: (index+1)*batch_size],
            x_g: dataset_x_g[index*batch_size: (index+1)*batch_size],
            y: dataset_y[index*batch_size: (index+1)*batch_size],
            av: dataset_av[index*batch_size: (index+1)*batch_size]
        }
    )

    # compile the theano function
    hessian = theano.function(
        inputs=[],
        outputs=rum.hessian(y),
        updates=None,
        on_unused_input='ignore',
        givens={
            x_ng: dataset_x_ng,
            x_g: dataset_x_g,
            y: dataset_y,
            av: dataset_av
        }
    )

    loglikelihood = theano.function(
        inputs=[],
        outputs=rum.loglikelihood(y),
        updates=None,
        on_unused_input='ignore',
        givens={
            x_ng: dataset_x_ng,
            x_g: dataset_x_g,
            y: dataset_y,
            av: dataset_av
        }
    )

    ##################
    # ESTIMATE MODEL #
    ##################

    best_loss = np.inf
    epoch = 0
    done_looping = False
    #calculate init loglikelihood
    nullLoglikelihood = loglikelihood()
    while epoch < n_epochs and done_looping is False:
        epoch += 1
        for minibatch_index in range(n_batches):
            estimation_loss = estimate_model(minibatch_index)
            print(estimation_loss)
            print(data_y)
            print('W_g=', rum.W_g.get_value(borrow=True))
            print('W_ng=', rum.W_ng.get_value(borrow=True))
            print('c=', rum.c.get_value(borrow=True))

        # when ASC[-1] reaches 0, stop
        if np.abs(rum.c.get_value(borrow=True)[-1]) <= 1e-5 and epoch >= 100:
           done_looping = True

    # calculate final loglikelihood
    finalLoglikelihood = loglikelihood()
    print('nullLoglikelihood=', nullLoglikelihood, 'finalLoglikelihood=', finalLoglikelihood)
    print('... solving Hessians')
    print([np.diagonal(mat) for mat in hessian()])
    serr = np.hstack([2/np.sqrt(n*np.diagonal(mat)) for mat in hessian()])
    #print(serr)

    param_names = ['ASC_Bus', 'ASC_CarRental', 'ASC_Car', 'ASC_Plane', 'ASC_TrH', 'ASC_Train', \
    'DrvLicens_Bus', 'DrvLicens_CarRental', 'DrvLicens_Car', 'DrvLicens_Plane', 'DrvLicens_TrH', 'DrvLicens_Train', \
    'PblcTrst_Bus', 'PblcTrst_CarRental', 'PblcTrst_Car', 'PblcTrst_Plane', 'PblcTrst_TrH', 'PblcTrst_Train', \
    'Ag1825_Bus', 'Ag1825_CarRental', 'Ag1825_Car', 'Ag1825_Plane', 'Ag1825_TrH', 'Ag1825_Train', \
    'Ag2545_Bus', 'Ag2545_CarRental', 'Ag2545_Car', 'Ag2545_Plane', 'Ag2545_TrH', 'Ag2545_Train', \
    'Ag4565_Bus', 'Ag4565_CarRental', 'Ag4565_Car', 'Ag4565_Plane', 'Ag4565_TrH', 'Ag4565_Train', \
    'Ag65M_Bus', 'Ag65M_CarRental', 'Ag65M_Car', 'Ag65M_Plane', 'Ag65M_TrH', 'Ag65M_Train', \
    'cost', 'tt', 'relib']
    betas = np.concatenate((rum.c.get_value(borrow=True), rum.W_ng.get_value(borrow=True), rum.W_g.get_value(borrow=True)), axis=0)
    t_stat = betas/serr
    df = pd.DataFrame(data=np.vstack((betas,t_stat)).T, index=param_names, columns=['betas', 't_stat'])
    # df.to_csv('mnl_estimation.csv')
    print(df)

if __name__ == '__main__':
    main()
