import pickle
import timeit
import numpy as np
import pandas as pd
import theano
import theano.tensor as T
from theano import shared, function, scan
from theano.tensor.shared_randomstreams import RandomStreams

from models import optimizers
from models.crbm import CRBM
from models.preprocessing import extractdata

""" Custom options """
floatX = theano.config.floatX
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.options.display.max_rows = 999
csvString = 'data/US_SP_Restructured.csv'


def run_crbm():
    """ Discrete choice model estimation with Theano

    Setup
    -----
    step 1: Load variables from csv file
    step 2: Define hyperparameters used in the computation
    step 3: define symbolic Theano tensors
    step 4: build model and define cost function
    step 5: define gradient calculation algorithm
    step 6: define Theano symbolic functions
    step 7: run main estimaiton loop for n iterations
    step 8: perform analytics and model statistics

    """
    # compile and import dataset from csv#
    d_x_ng, d_x_g, d_y, avail, d_ind = extractdata(csvString)
    data_x_ng = shared(np.asarray(d_x_ng, dtype=floatX), borrow=True)
    data_x_g = shared(np.asarray(d_x_g, dtype=floatX), borrow=True)
    data_y = T.cast(shared(np.asarray(d_y-1, dtype=floatX), borrow=True),
                    'int32')
    data_av = shared(np.asarray(avail, dtype=floatX), borrow=True)
    data_ind = shared(np.asarray(d_ind, dtype=floatX), borrow=True)

    sz_n = d_x_g.shape[0]  # number of samples
    sz_k = d_x_g.shape[1]  # number of generic variables
    sz_m = d_x_ng.shape[2]  # number of non-generic variables
    sz_i = d_x_ng.shape[1]  # number of alternatives
    sz_z = d_ind.shape[1]  # number of indicators

    sz_minibatch = sz_n  # model hyperparameters
    learning_rate = 0.1
    gen_rate = 1.0
    momentum = 0.9

    n_hidden = 3  # latent variable model parameters

    x_ng = T.tensor3('data_x_ng')  # symbolic theano tensors
    x_g = T.matrix('data_x_g')
    y = T.ivector('data_y')
    av = T.matrix('data_av')

    index = T.lscalar('index')

    z = T.matrix('data_ind')

    # construct model
    model = CRBM(
        sz_i, av,
        n_in=[(sz_m,), (sz_k, n_hidden)],
        n_hid=[(n_hidden,), (n_hidden, sz_i), (n_hidden, sz_z)],
        n_ind=(sz_z,),
        input=[x_ng, x_g],
        output=y,
        inds=z)

    cost, error, chain_end, updates = model.gibbs_sampling(
        y, x_ng, x_g, av, alts=6, steps=25)

    grads = T.grad(cost=cost-model.loglikelihood(y), wrt=model.params,
                   consider_constant=[chain_end])

    cost2 = - (model.loglikelihood(y) + 0.1*model.cross_entropy(z))

    grads2 = T.grad(cost=cost2, wrt=model.params2)

    opt = optimizers.adadelta(model.params, model.masks, momentum)
    opt2 = optimizers.adadelta(model.params2, model.masks2, momentum)
    # opt = optimizers.sgd(model.params, model.masks)

    updates.update(opt.updates(
        model.params, grads, learning_rate))

    updates2 = opt2.updates(model.params2, grads2, learning_rate)

    # null loglikelihood function
    fn_null = function(
        inputs=[],
        outputs=model.loglikelihood(y),
        givens={
            x_ng: data_x_ng,
            x_g: data_x_g,
            y: data_y,
            av: data_av},
        on_unused_input='ignore')

    # compile the theano functions
    fn_estimate = function(
        name='estimate',
        inputs=[index],
        outputs=[model.loglikelihood(y), cost],
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

    fn_optimize = function(
        name='optimize',
        inputs=[index],
        outputs=[model.loglikelihood(y)],
        updates=updates2,
        givens={
            x_ng: data_x_ng[
                index*sz_minibatch: T.min(((index+1)*sz_minibatch, sz_n))],
            x_g: data_x_g[
                index*sz_minibatch: T.min(((index+1)*sz_minibatch, sz_n))],
            y: data_y[
                index*sz_minibatch: T.min(((index+1)*sz_minibatch, sz_n))],
            av: data_av[
                index*sz_minibatch: T.min(((index+1)*sz_minibatch, sz_n))],
            z: data_ind[
                index*sz_minibatch: T.min(((index+1)*sz_minibatch, sz_n))]},
        allow_input_downcast=True,
        on_unused_input='ignore',)

    fn_pred = function(
        inputs=[],
        outputs=model.y_pred,
        givens={
            x_ng: data_x_ng,
            x_g: data_x_g,
            y: data_y,
            av: data_av},
        on_unused_input='ignore')

    """ Main estimation process loop """
    print('Begin estimation...')

    epoch = 0  # process loop parameters
    sz_epoches = 2000
    sz_batches = np.ceil(sz_n/sz_minibatch).astype(np.int32)
    done_looping = False
    patience = 300
    patience_inc = 10
    best_loglikelihood = -np.inf
    null_Loglikelihood = fn_null()
    start_time = timeit.default_timer()

    while epoch < sz_epoches and done_looping is False:
        epoch_cost = []
        epoch_loglikelihood = []
        for i in range(sz_batches):
            (batch_loglikelihood, batch_cost) = fn_estimate(i)
            epoch_cost.append(batch_cost)
            epoch_loglikelihood.append(batch_loglikelihood)

        this_loglikelihood = np.sum(epoch_loglikelihood)
        this_cost = np.sum(epoch_cost)
        print('@ iteration %d/%d loglikelihood: %.3f'
              % (epoch, patience, this_loglikelihood))
        print('               cost %.3f'
              % this_cost)
        print(fn_pred())
        print(data_y.eval())

        if this_loglikelihood > best_loglikelihood:
            if this_loglikelihood > 0.998 * best_loglikelihood:
                patience += patience_inc
            best_loglikelihood = this_loglikelihood
            best_model = model

        if (epoch > patience or
            this_loglikelihood < 1.01 * best_loglikelihood):
            done_looping = True

        epoch += 1

    epoch = 0
    patience = 900
    done_looping = False
    best_loglikelihood = -np.inf
    # done_looping = True
    while epoch < sz_epoches and done_looping is False:
        epoch_cost = []
        epoch_loglikelihood = []
        for i in range(sz_batches):
            (batch_loglikelihood) = fn_optimize(i)
            epoch_loglikelihood.append(batch_loglikelihood)

        this_loglikelihood = np.sum(epoch_loglikelihood)
        this_cost = np.sum(epoch_cost)
        print('@ iteration %d/%d loglikelihood: %.3f'
              % (epoch, patience, this_loglikelihood))
        print(fn_pred())
        print(data_y.eval())

        if this_loglikelihood > best_loglikelihood:
            if this_loglikelihood > 0.999 * best_loglikelihood:
                patience += patience_inc
            best_loglikelihood = this_loglikelihood
            best_model = model

        if (epoch > patience or
            this_loglikelihood < 1.01 * best_loglikelihood):
            done_looping = True

        epoch += 1

    final_Loglikelihood = best_loglikelihood
    rho_square = 1.-(final_Loglikelihood/null_Loglikelihood)

    with open('best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    end_time = timeit.default_timer()

    """ Analytics and model statistics """
    with open('best_model.pkl', 'rb') as f:
        best_model = pickle.load(f)

    print('... solving Hessians')
    # hessian function
    fn_hessian = function(
        inputs=[best_model.x_ng, best_model.x_g, best_model.av],
        outputs=T.hessian(
            cost=-(best_model.loglikelihood(y)+best_model.cross_entropy(z)),
            wrt=best_model.params2),
        givens={y: data_y, z: data_ind},
        on_unused_input='ignore')

    h = np.hstack([np.diagonal(mat) for mat in fn_hessian(data_x_ng.eval(), data_x_g.eval(), data_av.eval())])
    n_est_params = np.count_nonzero(h)
    aic = 2 * n_est_params - 2 * final_Loglikelihood
    bic = np.log(sz_n) * n_est_params - 2 * final_Loglikelihood

    print('@iteration %d, run time %.3f '
          % (epoch, end_time-start_time))
    print('Null Loglikelihood: %.3f'
          % null_Loglikelihood)
    print('Final Loglikelihood: %.3f'
          % final_Loglikelihood)
    print('rho square %.3f'
          % rho_square)
    print('AIC %.3f'
          % aic)
    print('BIC %.3f'
          % bic)


    run_analytics(best_model, h, n_hidden)


def run_analytics(model, hessians, n_latentVars):
    stderr = 2 / np.sqrt(hessians)
    betas = np.concatenate(
        [param.get_value() for param in model.params2], axis=0)
    t_stat = betas/stderr
    data = np.vstack((betas, stderr, t_stat)).T
    columns = ['betas', 'serr', 't_stat']

    paramNames = []  # print dataFrame

    choices = ['Bus', 'CarRental', 'Car', 'Plane', 'TrH', 'Train']

    ASCs = []
    for choice in choices:
        ASCs.append('ASC_'+choice)

    nongenericNames = ['cost', 'tt', 'relib']

    genericNames = [
		'DrvLicens', 'PblcTrst',
		'Ag1825', 'Ag2545', 'Ag4565', 'Ag65M',
		'Male', 'Fulltime', # 'PrtTime', 'Unemplyd',
		'Edu_Highschl', 'Edu_BSc', 'Edu_MscPhD',
		'HH_Veh0', 'HH_Veh1', 'HH_Veh2M',
		# 'HH_Adult1', 'HH_Adult2', 'HH_Adult3M',
		'HH_Chld0', 'HH_Chld1', 'HH_Chld2M',
		'HH_Inc020K', 'HH_Inc2060K', 'HH_Inc60KM',
		# 'HH_Sngl', 'HH_SnglParent', 'HH_AllAddults',
		# 'HH_Nuclear', # 'P_Chld',
		# 'O_MTL_US_max', 'O_Odr_US_max',
		# 'D_Bstn_max', 'D_NYC_max', 'D_Maine_max',
		# 'Tp_Onewy_max', 'Tp_2way_max',
		# 'Tp_h06_max', 'Tp_h69_max', 'Tp_h915_max',
		# 'Tp_h1519_max', 'Tp_h1924_max', 'Tp_h1524_max',
		# 'Tp_Y2016_max', 'Tp_Y2017_max',
		# 'Tp_Wntr_max', 'Tp_Sprng_max', 'Tp_Sumr_max', 'Tp_Fall_max',
		# 'Tp_CarDrv_max', 'Tp_CarPsngr_max', 'Tp_CarShrRnt_max',
		# 'Tp_Train_max', 'Tp_Bus_max', 'Tp_Plane_max', 'Tp_ModOdr_max',
		# 'Tp_WrkSkl_max', 'Tp_Leisr_max', 'Tp_Shpng_max',
		# 'Tp_ActOdr_max',
		# 'Tp_NHotel1_max', 'Tp_NHotel2_max', 'Tp_NHotel3M_max',
		# 'Tp_FreqMonthlMulti_max', 'Tp_FreqYearMulti_max',
		# 'Tp_FreqYear1_max',
	]

    latentNames = []
    latentConstants = []
    for n in np.arange(n_latentVars):
        latentNames.append('LV'+str(n))
        latentConstants.append('CONST_LV'+str(n))

    indicators = [
        'Envrn_Car', 'Envrn_Train', 'Envrn_Bus', 'Envrn_Plane',
        'Safe_Car', 'Safe_Train', 'Safe_Bus', 'Safe_Plane',
        'Comf_Car', 'Comf_Train', 'Comf_Bus', 'Comf_Plane'
    ]

    latentVariables = []
    for name in genericNames:
        for lv in latentNames:
            latentVariables.append(name+'_'+lv)

    indicatorVariables = []
    for lv in latentNames:
        for ind in indicators:
            indicatorVariables.append(lv+'_'+ind)

    for ASC in ASCs:
        paramNames.append(ASC)

    for name in nongenericNames:
        paramNames.append(name)

    for name in latentVariables:
        paramNames.append(name)

    # for name in latentConstants:
    #     paramNames.append(name)


    for name in latentNames:
        for choice in choices:
            paramNames.append(name+'_'+choice)

    for name in indicatorVariables:
        paramNames.append(name)

    df = pd.DataFrame(data, paramNames, columns)
    print(df)


if __name__ == '__main__':
    run_crbm()
