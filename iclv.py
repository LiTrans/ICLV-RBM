########################################
#
# @file bioestima.py
# @originalAuthor: Michel Bierlaire, EPFL
# @modified: Melvin Wong, Ryerson University
#
#######################################

from biogeme import *
from headers import *
from loglikelihood import *
from statistics import *

import numpy as np

#Parameters to be estimated
# Arguments:
#   - 1  Name for report; Typically, the same as the variable.
#   - 2  Starting value.
#   - 3  Lower bound.
#   - 4  Upper bound.
#   - 5  0: estimate the parameter, 1: keep it fixed.
#

labels = np.asarray(['_BUS', '_CARRENTAL', '_CAR', '_PLANE', '_TRH', '_TRAIN'])

# alternative specific parameters
ASC = [Beta('ASC'+str(label),0,-10,10,np.floor(i/5).astype(np.int),'ASC'+str(label)) for i, label in enumerate(labels)]

B_DRVLICENS = [Beta('B_DRVLICENS'+str(label),0,-10,10,np.floor(i/5).astype(np.int),'B_DRVLICENS'+str(label)) for i, label in enumerate(labels)]

B_PBLCTRST = [Beta('B_PBLCTRST'+str(label),0,-10,10,np.floor(i/5).astype(np.int),'B_PBLCTRST'+str(label)) for i, label in enumerate(labels)]
#
# B_AG1825 = [Beta('B_AG1825'+str(label),0,-10,10,np.floor(i/5).astype(np.int),'B_AG1825'+str(label)) for i, label in enumerate(labels)]
#
# B_AG2545 = [Beta('B_AG2545'+str(label),0,-10,10,np.floor(i/5).astype(np.int),'B_AG2545'+str(label)) for i, label in enumerate(labels)]
#
# B_AG4565 = [Beta('B_AG4565'+str(label),0,-10,10,np.floor(i/5).astype(np.int),'B_AG4565'+str(label)) for i, label in enumerate(labels)]
#
# B_AG65M = [Beta('B_AG65M'+str(label),0,-10,10,np.floor(i/5).astype(np.int),'B_AG65M'+str(label)) for i, label in enumerate(labels)]
#
# B_MALE = [Beta('B_MALE'+str(label),0,-10,10,np.floor(i/5).astype(np.int),'B_MALE'+str(label)) for i, label in enumerate(labels)]
#
# B_FULLTIME = [Beta('B_FULLTIME'+str(label),0,-10,10,np.floor(i/5).astype(np.int),'B_FULLTIME'+str(label)) for i, label in enumerate(labels)]
#
# B_PRTTIME = [Beta('B_PRTTIME'+str(label),0,-10,10,np.floor(i/5).astype(np.int),'B_PRTTIME'+str(label)) for i, label in enumerate(labels)]
#
# B_UNEMPLYD = [Beta('B_UNEMPLYD'+str(label),0,-10,10,np.floor(i/5).astype(np.int),'B_UNEMPLYD'+str(label)) for i, label in enumerate(labels)]
#
# B_EDU_HIGHSCHL = [Beta('B_EDU_HIGHSCHL'+str(label),0,-10,10,np.floor(i/5).astype(np.int),'B_EDU_HIGHSCHL'+str(label)) for i, label in enumerate(labels)]
#
# B_EDU_BSC = [Beta('B_EDU_BSC'+str(label),0,-10,10,np.floor(i/5).astype(np.int),'B_EDU_BSC'+str(label)) for i, label in enumerate(labels)]
#
# B_EDU_MSCPHD = [Beta('B_EDU_MSCPHD'+str(label),0,-10,10,np.floor(i/5).astype(np.int),'B_EDU_MSCPHD'+str(label)) for i, label in enumerate(labels)]


# generic parameters
B_COST = Beta('B_COST',0,-10,10,0,'B_COST')
B_TIME = Beta('B_TIME',0,-10,10,0,'B_TIME')
B_RELI = Beta('B_RELI',0,-10,10,0,'B_RELI')
B_COST_S = Beta('B_COST_S',0,-10,10,0,'B_COST_S')
B_TIME_S = Beta('B_TIME_S',0,-10,10,0,'B_TIME_S')
B_RELI_S = Beta('B_RELI_S',0,-10,10,0,'B_RELI_S')

BIOGEME_OBJECT.DRAWS = {
    'B_COST_RND': 'NORMAL',
    'B_TIME_RND': 'NORMAL',
    'B_RELI_RND': 'NORMAL'}

B_COST_RND = B_COST + B_COST_S * bioDraws('B_COST_RND')
B_TIME_RND = B_TIME + B_TIME_S * bioDraws('B_TIME_RND')
B_RELI_RND = B_RELI + B_RELI_S * bioDraws('B_RELI_RND')

altspec_params = np.asarray([B_DRVLICENS, B_PBLCTRST])#, B_AG1825, B_AG2545, B_AG4565, B_AG65M, B_MALE, B_FULLTIME, B_PRTTIME, B_UNEMPLYD, B_EDU_HIGHSCHL, B_EDU_BSC, B_EDU_MSCPHD])
gen_params = np.asarray([B_COST, B_TIME, B_RELI])

# generic variables
gen_vars = np.asarray([DrvLicens, PblcTrst])#, Ag1825, Ag2545, Ag4565, Ag65M, Male, Fulltime, PrtTime, Unemplyd, Edu_Highschl, Edu_BSc, Edu_MscPhD])

# alternative specific variables
bus = [Bus_Cost/100., Bus_TT/100., BusRelib/100.]
carrental = [CarRental_Cost/100., CarRental_TT/100., CarRentalRelib/100.]
car = [Car_Cost/100., Car_TT/100., CarRelib/100.]
plane = [Plane_Cost/100., Plane_TT/100., PlaneRelib/100.]
trh = [TrH_Cost/100., TrH_TT/100., TrHRelib/100.]
train = [Train_Cost/100., Train_TT/100., TrainRelib/100.]

altspec_vars = np.asarray([bus, carrental, car, plane, trh, train])

# Utility functions
# For numerical reasons, it is good practice to scale the data to
# that the values of the parameters are around 1.0.
# A previous estimation with the unscaled data has generated
# parameters around -0.01 for both cost and time. Therefore, time and
# cost are multipled by 0.01.

V1 = ASC[0] + np.dot(gen_params, altspec_vars[0]) + np.dot(gen_vars, altspec_params[:,0])
V2 = ASC[1] + np.dot(gen_params, altspec_vars[1]) + np.dot(gen_vars, altspec_params[:,1])
V3 = ASC[2] + np.dot(gen_params, altspec_vars[2]) + np.dot(gen_vars, altspec_params[:,2])
V4 = ASC[3] + np.dot(gen_params, altspec_vars[3]) + np.dot(gen_vars, altspec_params[:,3])
V5 = ASC[4] + np.dot(gen_params, altspec_vars[4]) + np.dot(gen_vars, altspec_params[:,4])
V6 = ASC[5] + np.dot(gen_params, altspec_vars[5]) + np.dot(gen_vars, altspec_params[:,5])

# Associate utility functions with the numbering of alternatives
V = {1: V1, 2: V2, 3: V3, 4: V4, 5: V5, 6: V6}

# Associate the availability conditions with the alternatives
av = {1: AV_Bus,
      2: AV_CarRental,
      3: AV_Car,
      4: AV_Plane,
      5: AV_TrH,
      6: AV_Train}

# The choice model is a logit, with availability conditions
logprob = bioLogLogit(V, av, New_SP_Choice)
# Defines an itertor on the data
rowIterator('obsIter')
# DEfine the likelihood function for the estimation
BIOGEME_OBJECT.ESTIMATE = Sum(logprob,'obsIter')

# prob = bioLogit(V, av, New_SP_Choice)
# l = mixedloglikelihood(prob)
# rowIterator('obsIter')
# # Likelihood function
# BIOGEME_OBJECT.ESTIMATE = Sum(l,'obsIter')

# All observations verifying the following expression will not be
# considered for estimation
# The modeler here has developed the model only for work trips.
# Observations such that the dependent variable CHOICE is 0 are also removed.
# exclude = (( PURPOSE != 1 ) * (  PURPOSE   !=  3  ) + ( CHOICE == 0 )) > 0

# BIOGEME_OBJECT.EXCLUDE = exclude

# Statistics

nullLoglikelihood(av,'obsIter')
choiceSet = [1, 2, 3, 4, 5, 6]
cteLoglikelihood(choiceSet, New_SP_Choice, 'obsIter')
availabilityStatistics(av, 'obsIter')

BIOGEME_OBJECT.PARAMETERS['optimizationAlgorithm'] = "BIO"
BIOGEME_OBJECT.PARAMETERS['numberOfThreads'] = "4"
BIOGEME_OBJECT.PARAMETERS['NbrOfDraws'] = "25"

BIOGEME_OBJECT.FORMULAS['Bus utility'] = V1
BIOGEME_OBJECT.FORMULAS['CarRental utility'] = V2
BIOGEME_OBJECT.FORMULAS['Car utility'] = V3
