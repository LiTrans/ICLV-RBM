import pickle
import timeit
import numpy as np
import pandas as pd
import theano
import theano.tensor as T
from theano import shared, function
from theano.tensor.shared_randomstreams import RandomStreams

from dataWrangling import *
from Logistic import *
from optimizers import *

""" Custom options """
floatX = theano.config.floatX
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.options.display.max_rows = 999


def main():
    """ Discrete choice model estimation with theano
        Under normal circumstances, no modification
        necessary

    Setup
    -----
    step 1: Load variables from csv file
    step 2: Define hyperparameters used in the computation
    step 3: define symbolic Theano tensors
    step 4: build model and define cost function
    step 5: define gradient calculation algorithm
    step 6: define Theano symbolic functions

    """
    # compile and import dataset from csv#
    dataset = Preprocessing('US_SP_Restructured.csv')
    dataset_x_ng, dataset_x_g, dataset_y, avail, data_ind = dataset.data()
    data_x_ng = shared(np.asarray(dataset_x_ng, dtype=floatX), borrow=True)
    data_x_g = shared(np.asarray(dataset_x_g, dtype=floatX), borrow=True)
    data_y = T.cast(
        shared(np.asarray(dataset_y-1, dtype=floatX), borrow=True),
        'int32')
    data_av = shared(np.asarray(avail, dtype=floatX), borrow=True)

    sz_n = dataset_x_g.shape[0]  # number of samples
    sz_k = dataset_x_g.shape[1]  # number of generic variables
    sz_m = dataset_x_ng.shape[2]  # number of non-generic variables
    sz_i = dataset_x_ng.shape[1]  # number of alternatives

    sz_minibatch = sz_n  # model hyperparameters
    learning_rate = 0.3
    momentum = 0.9

    x_ng = T.tensor3('data_x_ng')  # symbolic theano tensors
    x_g = T.matrix('data_x_g')
    y = T.ivector('data_y')
    av = T.matrix('data_av')

    index = T.lscalar('index')

    # construct model
    model = Logistic(
        sz_i, av, input=[x_ng, x_g], n_in=[(sz_m,), (sz_k, sz_i)])

    cost = -model.loglikelihood(y)

    # calculate the gradients wrt to the loss function
    grads = T.grad(cost=cost, wrt=model.params)
    optimizer = adadelta(model.params, model.masks, momentum)

    updates = optimizer.updates(
        model.params, grads, learning_rate)

    # hessian function
    fn_hessian = function(
        inputs=[],
        outputs=T.hessian(cost=cost, wrt=model.params),
        givens={
            x_ng: data_x_ng,
            x_g: data_x_g,
            y: data_y,
            av: data_av})

    # null loglikelihood function
    fn_null = function(
        inputs=[],
        outputs=-model.loglikelihood(y),
        givens={
            x_ng: data_x_ng,
            x_g: data_x_g,
            y: data_y,
            av: data_av})

    # compile the theano functions
    fn_estimate = function(
        name='estimate',
        inputs=[index],
        outputs=[model.loglikelihood(y), model.errors(y)],
        updates=updates,
        givens={
            x_ng: data_x_ng[
                index*sz_minibatch: T.min(((index+1)*sz_minibatch, sz_n))],
            x_g: data_x_g[
                index*sz_minibatch: T.min(((index+1)*sz_minibatch, sz_n))],
            y: data_y[
                index*sz_minibatch: T.min(((index+1)*sz_minibatch, sz_n))],
            av: data_av[
                index*sz_minibatch: T.min(((index+1)*sz_minibatch, sz_n))]},
        allow_input_downcast=True,
        on_unused_input='ignore',)

    """ Main estimation process loop """
    print('Begin estimation...')

    epoch = 0  # process loop parameters
    sz_epoches = 9999
    sz_batches = np.ceil(sz_n/sz_minibatch).astype(np.int32)
    done_looping = False
    patience = 300
    patience_inc = 10
    best_loglikelihood = -np.inf
    null_Loglikelihood = -fn_null()
    start_time = timeit.default_timer()

    while epoch < sz_epoches and done_looping is False:
        epoch_error = []
        epoch_loglikelihood = []
        for i in range(sz_batches):
            (batch_loglikelihood, batch_error) = fn_estimate(i)
            epoch_error.append(batch_error)
            epoch_loglikelihood.append(batch_loglikelihood)

        this_loglikelihood = np.sum(epoch_loglikelihood)
        print('@ iteration %d loglikelihood: %.3f'
              % (epoch, this_loglikelihood))

        if this_loglikelihood > best_loglikelihood:
            if this_loglikelihood > 0.995 * best_loglikelihood:
                patience += patience_inc
            best_loglikelihood = this_loglikelihood
            with open('best_model.pkl', 'wb') as f:
                pickle.dump(model, f)

        if epoch > patience:
            done_looping = True

        epoch += 1

    final_Loglikelihood = best_loglikelihood
    rho_square = 1.-(final_Loglikelihood/null_Loglikelihood)
    end_time = timeit.default_timer()

    """ Analytics and model statistics """
    print('@iteration %d, run time %.3f '
          % (epoch, end_time-start_time))
    print('Null Loglikelihood: %.3f'
          % null_Loglikelihood)
    print('Final Loglikelihood: %.3f'
          % final_Loglikelihood)
    print('rho square %.3f'
          % rho_square)

    with open('best_model.pkl', 'rb') as f:
        best_model = pickle.load(f)

    print('... solving Hessians')
    h = np.hstack([np.diagonal(mat) for mat in fn_hessian()])
    run_analytics(best_model, h)


def run_analytics(model, hessians):
    stderr = 2 / np.sqrt(hessians)
    betas = np.concatenate(
        [param.get_value() for param in model.params], axis=0)
    t_stat = betas/stderr
    data = np.vstack((betas, stderr, t_stat)).T
    columns = ['betas', 'serr', 't_stat']

    paramNames = []  # print dataFrame

    choices = [
        'Bus', 'CarRental', 'Car', 'Plane', 'TrH', 'Train'
    ]
    ASCs = [
        'ASC_Bus', 'ASC_CarRental', 'ASC_Car', 'ASC_Plane', 'ASC_TrH',
        'ASC_Train'
    ]
    nongenericNames = [
        'cost', 'tt', 'relib',  # 'cost_s', 'tt_s', 'relib_s'
    ]
    genericNames = [
        'DrvLicens', 'PblcTrst',
        # 'Ag1825', 'Ag2545', 'Ag4565', 'Ag65M', 'Male', 'Fulltime',
        # 'PrtTime', 'Unemplyd', 'Edu_Highschl', 'Edu_BSc', 'Edu_MscPhD',
        # 'HH_Veh0', 'HH_Veh1',
        # 'HH_Veh2M', 'HH_Adult1', 'HH_Adult2', 'HH_Adult3M', 'HH_Chld0',
        # 'HH_Chld1', 'HH_Chld2M',
        # 'HH_Inc020K', 'HH_Inc2060K', 'HH_Inc60KM', 'HH_Sngl',
        # 'HH_SnglParent', 'HH_AllAddults', 'HH_Nuclear', 'P_Chld',
        # 'O_MTL_US_max', 'O_Odr_US_max', 'D_Bstn_max',
        # 'D_NYC_max', 'D_Maine_max',
        # 'Tp_Onewy_max', 'Tp_2way_max', 'Tp_h06_max', 'Tp_h69_max',
        # 'Tp_h915_max', 'Tp_h1519_max', 'Tp_h1924_max', 'Tp_h1524_max',
        # 'Tp_Y2016_max', 'Tp_Y2017_max',
        # 'Tp_Wntr_max', 'Tp_Sprng_max', 'Tp_Sumr_max', 'Tp_Fall_max',
        # 'Tp_CarDrv_max', 'Tp_CarPsngr_max', 'Tp_CarShrRnt_max',
        # 'Tp_Train_max', 'Tp_Bus_max',
        # 'Tp_Plane_max', 'Tp_ModOdr_max', 'Tp_WrkSkl_max', 'Tp_Leisr_max',
        # 'Tp_Shpng_max', 'Tp_ActOdr_max',
        # 'Tp_NHotel1_max', 'Tp_NHotel2_max', 'Tp_NHotel3M_max',
        # 'Tp_FreqMonthlMulti_max', 'Tp_FreqYearMulti_max'
    ]

    for ASC in ASCs:
        paramNames.append(ASC)
    for name in nongenericNames:
        paramNames.append(name)
    for name in genericNames:
        for choice in choices:
            paramNames.append(name+'_'+choice)

    df = pd.DataFrame(data, paramNames, columns)
    print(df)


if __name__ == '__main__':
    main()
