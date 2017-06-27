import numpy as np
import pandas as pd
import theano, os, sys, timeit
import theano.tensor as T

from theano import shared
from dataWrangling import *
from optimizers import *

class RandomUtilityModel(object):
    """ Discrete choice model class structure"""
    def __init__(self, n_in_ng, n_in_g, n_out, av, genericInput=None, nongenericInput=None):
        # initialize bias c of shape (i,) (Alternative specific constants)
        #self.c = [shared(value=np.zeros((1,), dtype=theano.config.floatX), name='c' + str(i), borrow=True) for i in range(n_out)]

        # add to utility
        #self.utility = T.stack(self.c, axis=1).reshape((n_out,))
        self.c = shared(value=np.zeros((n_out,), dtype=theano.config.floatX), name='c', borrow=True)
        self.utility = self.c
        self.params = []

        # construct parameter values for generic and non-generic inputs
        # non-generic paramters for generic variables
        if genericInput is not None:
            # self.W_ng = shared(value=np.zeros((n_in_g, n_out), dtype=theano.config.floatX), name='W_ng', borrow=True)
            self.W_ng_flat = shared(value=np.zeros((n_in_g * n_out), dtype=theano.config.floatX), name='W_ng_flat', borrow=True)
            self.W_ng = self.W_ng_flat.reshape((n_in_g, n_out))
            genericUtility = self.linear(genericInput, self.W_ng)

            if self.utility is None:
                self.utility = genericUtility
            else:
                self.utility += genericUtility
            self.params.extend([self.W_ng_flat])

        if nongenericInput is not None:
            self.W_g = shared(value=np.zeros((n_in_ng,), dtype=theano.config.floatX), name='W_g', borrow=True)
            nongenericUtility = self.linear(nongenericInput, self.W_g)

            if self.utility is None:
                self.utility = nongenericUtility
            else:
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
        self.av = av
        self.params.extend([self.c])

    def linear(self, input, params):
        # performs a dot (inner) product of input and parameters
        return T.dot(input, params)

    def softmax(self, x, av):
        e_x = av*T.exp(x - x.max(axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def loglikelihood(self, y):
        # full loglikelihood
        return T.sum(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        # returns the number of errors as a percentage of total number of examples
        return T.mean(T.neq(self.y_pred, y))

    def hessian(self, y):
        # returns the hessian of the negative loglikelihood wrt parameters
        hessian_ll = -T.sum(T.log(T.switch(T.eq(self.p_y_given_x, 0), 1, self.p_y_given_x))[T.arange(y.shape[0]), y])
        return T.hessian(cost=hessian_ll, wrt=self.params)

def main():
    # import dataset
    data_x_ng, data_x_g, data_y, availability = dataWrangling()
    dataset_x_ng = shared(np.asarray(data_x_ng, dtype=theano.config.floatX), borrow=True)
    dataset_x_g = shared(np.asarray(data_x_g, dtype=theano.config.floatX), borrow=True)
    dataset_y = T.cast(shared(np.asarray(data_y-1, dtype=theano.config.floatX), borrow=True), 'int32')
    dataset_av = shared(np.asarray(availability, dtype=theano.config.floatX), borrow=True)

    # check size of array
    n, i, m = data_x_ng.shape # (rows, choices, variables.nongenerics)
    _, k = data_x_g.shape # (variables.generic)

    # hyperparameters
    minibatch_size = n #n
    lr = 1e-3

    # generate symbolic variables for inputs and output
    x_ng = T.tensor3('x_ng')
    x_g = T.matrix('x_g')
    y = T.ivector('y')
    av = T.matrix('av')

    # allocate symobolic variable to index
    index = T.lscalar()

    # construct symbolic representation of the Logit function
    rum = RandomUtilityModel(n_in_ng=m, n_in_g=k, n_out=i, av=av, genericInput=x_g, nongenericInput=x_ng)

    # define the cost function
    cost = -rum.loglikelihood(y)

    # obtaining the gradients wrt to the loss function
    grads = T.grad(cost=cost, wrt=rum.params)

    optimizer = rmsprop(rum.params)
    # updates = optimizer.updates(params=rum.params, grads=grads, learning_rate=lr)
    updates = optimizer.updates(params=rum.params, grads=grads, learning_rate=lr, momentum=0.95)

    # compile the theano function
    estimate_model = theano.function(
        inputs=[index],
        outputs=[rum.errors(y), rum.loglikelihood(y)],
        updates=updates,
        on_unused_input='ignore',
        givens={
            x_ng: dataset_x_ng[index*minibatch_size: T.min((n, (index+1)*minibatch_size))],
            x_g: dataset_x_g[index*minibatch_size: T.min((n, (index+1)*minibatch_size))],
            y: dataset_y[index*minibatch_size: T.min((n, (index+1)*minibatch_size))],
            av: dataset_av[index*minibatch_size: T.min((n, (index+1)*minibatch_size))]
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

    # initialize estimation parameters
    n_batches = n // minibatch_size
    best_loss = np.inf
    epoch = 0
    n_epochs = 7500
    done_looping = False
    patience = 100 # look at these many iterations
    patience_increase = 20
    start_time = timeit.default_timer() # start timer

    #calculate init loglikelihood
    nullNegLoglikelihood = -loglikelihood()

    while epoch < n_epochs and done_looping is False:

        epoch_error = []
        epoch_loglikelihood = []
        epoch += 1

        # iterate over all samples
        for minibatch_index in range(n_batches):
            (batch_error, batch_loglikelihood) = estimate_model(minibatch_index) # do gradient updates
            epoch_error.append(batch_error)
            epoch_loglikelihood.append(batch_loglikelihood)

        print('prediction error: %.3f%%' % (np.mean(epoch_error)*100))
        print('loglikelihood: %.3f' % np.mean(epoch_loglikelihood))
        print('W_ng=', rum.W_ng.eval())
        print('W_g=', rum.W_g.eval())
        print('c=', rum.c.eval())

        this_loss = -np.mean(epoch_loglikelihood)/n # average loglikelihood per sample
        if this_loss < best_loss:
            if this_loss <= 0.998 * best_loss:
                patience += patience_increase
            best_loss = this_loss

        if epoch > patience:
            done_looping = True

    end_time = timeit.default_timer() # end timer

    # calculate final loglikelihood
    finalNegLoglikelihood = -loglikelihood()
    print('@iteration %d, run time %.3f ' % (epoch, end_time-start_time))
    print('nullLoglikelihood: %.3f' % nullNegLoglikelihood)
    print('finalLoglikelihood: %.3f' % finalNegLoglikelihood)
    print('... solving Hessians', [np.diagonal(mat).shape for mat in hessian()])
    serr = np.hstack([2/np.sqrt(np.diagonal(mat)) for mat in hessian()])

    # param_names = ['ASC_Bus', 'ASC_CarRental', 'ASC_Car', 'ASC_Plane', 'ASC_TrH', 'ASC_Train', \
    # 'DrvLicens_Bus', 'DrvLicens_CarRental', 'DrvLicens_Car', 'DrvLicens_Plane', 'DrvLicens_TrH', 'DrvLicens_Train', \
    # 'PblcTrst_Bus', 'PblcTrst_CarRental', 'PblcTrst_Car', 'PblcTrst_Plane', 'PblcTrst_TrH', 'PblcTrst_Train', \
    # 'Ag1825_Bus', 'Ag1825_CarRental', 'Ag1825_Car', 'Ag1825_Plane', 'Ag1825_TrH', 'Ag1825_Train', \
    # 'Ag2545_Bus', 'Ag2545_CarRental', 'Ag2545_Car', 'Ag2545_Plane', 'Ag2545_TrH', 'Ag2545_Train', \
    # 'Ag4565_Bus', 'Ag4565_CarRental', 'Ag4565_Car', 'Ag4565_Plane', 'Ag4565_TrH', 'Ag4565_Train', \
    # 'Ag65M_Bus', 'Ag65M_CarRental', 'Ag65M_Car', 'Ag65M_Plane', 'Ag65M_TrH', 'Ag65M_Train', \
    # 'cost', 'tt', 'relib']

    param_names = [
    'DrvLicens_Bus', 'DrvLicens_CarRental', 'DrvLicens_Car', 'DrvLicens_Plane', 'DrvLicens_TrH', 'DrvLicens_Train', \
    'PblcTrst_Bus', 'PblcTrst_CarRental', 'PblcTrst_Car', 'PblcTrst_Plane', 'PblcTrst_TrH', 'PblcTrst_Train', \
    'Ag1825_Bus', 'Ag1825_CarRental', 'Ag1825_Car', 'Ag1825_Plane', 'Ag1825_TrH', 'Ag1825_Train', \
    'Ag2545_Bus', 'Ag2545_CarRental', 'Ag2545_Car', 'Ag2545_Plane', 'Ag2545_TrH', 'Ag2545_Train', \
    'Male_Bus', 'Male_CarRental', 'Male_Car', 'Male_Plane', 'Male_TrH', 'Male_Train', \
    'cost', 'tt', 'relib', \
    'ASC_Bus', 'ASC_CarRental', 'ASC_Car', 'ASC_Plane', 'ASC_TrH', 'ASC_Train'
    ]
    # betas = np.concatenate((rum.c.get_value(borrow=True), rum.W_ng.get_value(borrow=True), rum.W_g.get_value(borrow=True)), axis=0)
    betas = np.concatenate((rum.W_ng_flat.get_value(borrow=True), rum.W_g.get_value(borrow=True), rum.c.get_value(borrow=True)), axis=0)
    t_stat = betas/serr
    df = pd.DataFrame(data=np.vstack((betas,serr, betas/serr)).T, index=param_names, columns=['betas', 'serr', 't-test'])
    # df.to_csv('mnl_estimation.csv')
    print(df)

if __name__ == '__main__':
    main()
