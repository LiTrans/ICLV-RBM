import numpy as np
import pandas as pd
import theano
import theano.tensor as T

from theano import shared
from dataWrangling import *

class Logit(object):
    """ Basic non-generic variable Logit class """
    def __init__(self, input, n_in, n_out, av):
        # initialize weights W of shape (m,)
        self.W = shared(value=np.zeros(n_in, dtype=theano.config.floatX), name='W', borrow=True)

        # initialize bias c of shape (i,) (Alternative specific constants)
        self.c = shared(value=np.zeros(n_out, dtype=theano.config.floatX), name='c', borrow=True)

        # symbolic expression for computing the matrix of choice probabilities
        # each row prior to the softmax function is the utilities of each choice i
        out = T.dot(input, self.W) + self.c
        self.p_y_given_x = self.softmax(out, av)

        # compute prediction where choice is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # keep track of input
        self.input = input
        self.av = av

    def softmax(self, x, av):
        e_x = av*T.exp(x - x.max(axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True) + 1e-8

    def negLogLikelihood(self, y):
        # define the loss function
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        # returns the number of errors as a percentage of total number of examples
        return T.mean(T.neq(self.y_pred, y))

def main():
    # import dataset
    data_x, data_y, availability = dataWrangling()
    dataset_x = shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=True)
    dataset_y = T.cast(shared(np.asarray(data_y-1, dtype=theano.config.floatX), borrow=True), 'int32')
    dataset_av = shared(np.asarray(availability, dtype=theano.config.floatX), borrow=True)

    # check size of array
    n, i, m = data_x.shape

    # hyperparameters
    batch_size = n
    n_batches = 1
    n_epochs = 5000

    # generate symbolic variables for input and output
    x = T.tensor3('x')
    y = T.ivector('y')
    av = T.matrix('av')

    # allocate symobolic variable to index
    index = T.lscalar()

    # construct symbolic representation of the Logit function
    logit = Logit(input=x, n_in=m, n_out=i, av=av)
    cost = logit.negLogLikelihood(y)

    # obtaining the gradients wrt to the loss function
    g_W = T.grad(cost=cost, wrt=logit.W)
    g_c = T.grad(cost=cost, wrt=logit.c)
    lr = 0.2

    updates = [(logit.W, logit.W-lr*g_W),
               (logit.c, logit.c-lr*g_c)]

    # compile the theano function
    estimate_model = theano.function(
        inputs=[index],
        outputs=logit.errors(y),
        updates=updates,
        on_unused_input='ignore',
        givens={
            x: dataset_x[index*batch_size: (index+1)*batch_size],
            y: dataset_y[index*batch_size: (index+1)*batch_size],
            av: dataset_av[index*batch_size: (index+1)*batch_size]
        }
    )

    # compile the theano function
    hessian = theano.function(
        inputs=[index],
        outputs=T.hessian(cost=cost, wrt=[logit.W, logit.c]),
        updates=None,
        on_unused_input='ignore',
        givens={
            x: dataset_x[index*batch_size: (index+1)*batch_size],
            y: dataset_y[index*batch_size: (index+1)*batch_size],
            av: dataset_av[index*batch_size: (index+1)*batch_size]
        }
    )

    loglikelihood = theano.function(
        inputs=[],
        outputs=T.sum(T.log(logit.p_y_given_x)[T.arange(y.shape[0]), y]),
        updates=None,
        on_unused_input='ignore',
        givens={
            x: dataset_x,
            y: dataset_y,
            av: dataset_av
        }
    )

    ##################
    # ESTIMATE MODEL #
    ##################

    best_loss = np.inf
    epoch = 0

    #calculate init loglikelihood
    nullLoglikelihood = loglikelihood()
    while epoch < n_epochs:
        epoch += 1
        for minibatch_index in range(n_batches):
            estimation_loss = estimate_model(minibatch_index)
            print(estimation_loss)
            print(data_y)
            print('W=', logit.W.get_value(borrow=True))
            print('c=', logit.c.get_value(borrow=True))

    # calculate final loglikelihood
    finalLoglikelihood = loglikelihood()
    print('nullLoglikelihood=', nullLoglikelihood, 'finalLoglikelihood=', finalLoglikelihood)
    print('... solving Hessians')
    serr = np.hstack([2/np.sqrt(n*np.diagonal(mat)) for mat in hessian(0)])
    print(serr)

    param_names = ['cost', 'tt', 'relib', 'ASC_bus', 'ASC_CarRental', 'ASC_Car', 'ASC_Plane', 'ASC_TrH', 'ASC_Train']
    betas = np.concatenate((logit.W.get_value(borrow=True), logit.c.get_value(borrow=True)), axis=0)
    t_stat = betas/serr
    df = pd.DataFrame(data=np.vstack((betas,t_stat)).T, index=param_names, columns=['betas', 't_stat'])
    # df.to_csv('mnl_estimation.csv')
    print(df)

if __name__ == '__main__':
    main()
