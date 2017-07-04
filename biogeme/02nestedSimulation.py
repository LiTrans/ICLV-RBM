## File 02nestedSimulation.py
## Simple nested logit model for the Optima case study
## Wed May 10 11:24:32 2017

from biogeme import *
from headers import *
from statistics import *
from nested import *

### Three alternatives:
# CAR: automobile
# PT: public transportation
# SM: slow mode (walking, biking)

### List of parameters and their estimated value.
ASC_CAR = Beta('ASC_CAR',0.261291,-10000,10000,0,'ASC_CAR' )
ASC_SM = Beta('ASC_SM',0.0590204,-10000,10000,0,'ASC_SM' )
BETA_TIME_FULLTIME = \
 Beta('BETA_TIME_FULLTIME',-1.59709,-10000,10000,0,'BETA_TIME_FULLTIME' )
BETA_TIME_OTHER = \
 Beta('BETA_TIME_OTHER',-0.577362,-10000,10000,0,'BETA_TIME_OTHER' )
BETA_DIST_MALE = \
 Beta('BETA_DIST_MALE',-0.686327,-10000,10000,0,'BETA_DIST_MALE' )
BETA_DIST_FEMALE = \
 Beta('BETA_DIST_FEMALE',-0.83121,-10000,10000,0,'BETA_DIST_FEMALE' )
BETA_DIST_UNREPORTED = \
 Beta('BETA_DIST_UNREPORTED',-0.702974,-10000,10000,0,'BETA_DIST_UNREPORTED' )
BETA_COST = \
 Beta('BETA_COST',-0.716192,-10000,10000,0,'BETA_COST' )


###Definition of variables:
# For numerical reasons, it is good practice to scale the data to
# that the values of the parameters are around 1.0.

# The following statements are designed to preprocess the data. It is
# like creating a new columns in the data file. This should be
# preferred to the statement like
# TimePT_scaled = Time_PT / 200.0
# which will cause the division to be reevaluated again and again,
# throuh the iterations. For models taking a long time to estimate, it
# may make a significant difference.

TimePT_scaled = DefineVariable('TimePT_scaled', TimePT / 200 )
TimeCar_scaled = DefineVariable('TimeCar_scaled', TimeCar / 200 )
MarginalCostPT_scaled = DefineVariable('MarginalCostPT_scaled',
                                       MarginalCostPT / 10 )
CostCarCHF_scaled = DefineVariable('CostCarCHF_scaled',
                                   CostCarCHF / 10 )
distance_km_scaled = DefineVariable('distance_km_scaled',
                                    distance_km / 5 )

male = DefineVariable('male',Gender == 1)
female = DefineVariable('female',Gender == 2)
unreportedGender  = DefineVariable('unreportedGender',Gender == -1)

fulltime = DefineVariable('fulltime',OccupStat == 1)
notfulltime = DefineVariable('notfulltime',OccupStat != 1)

### Definition of utility functions:
V_PT = BETA_TIME_FULLTIME * TimePT_scaled * fulltime + \
       BETA_TIME_OTHER * TimePT_scaled * notfulltime + \
       BETA_COST * MarginalCostPT_scaled
V_CAR = ASC_CAR + \
        BETA_TIME_FULLTIME * TimeCar_scaled * fulltime + \
        BETA_TIME_OTHER * TimeCar_scaled * notfulltime + \
        BETA_COST * CostCarCHF_scaled
V_SM = ASC_SM + \
       BETA_DIST_MALE * distance_km_scaled * male + \
       BETA_DIST_FEMALE * distance_km_scaled * female + \
       BETA_DIST_UNREPORTED * distance_km_scaled * unreportedGender


# Associate utility functions with the numbering of alternatives
V = {0: V_PT,
     1: V_CAR,
     2: V_SM}

# Associate the availability conditions with the alternatives.
# In this example all alternatives are available for each individual.
av = {0: 1,
      1: 1,
      2: 1}

### DEFINITION OF THE NESTS:
# 1: nests parameter
# 2: list of alternatives

NEST_NOCAR = Beta('NEST_NOCAR',1.52853,1,10,0,'NEST_NOCAR' )


CAR = 1.0 , [ 1]
NO_CAR = NEST_NOCAR , [ 0,  2]
nests = CAR, NO_CAR

# All observations verifying the following expression will not be
# considered for estimation
exclude = (Choice   ==  -1)
BIOGEME_OBJECT.EXCLUDE =  exclude

##
## This has been copied-pasted from the file 01nestedEstimation_param.py
##
## Code for the sensitivity analysis generated after the estimation of the model
names = ['ASC_CAR','ASC_SM','BETA_COST','BETA_DIST_FEMALE','BETA_DIST_MALE','BETA_DIST_UNREPORTED','BETA_TIME_FULLTIME','BETA_TIME_OTHER','NEST_NOCAR']
values = [[0.0100225,-0.0023271,0.00151986,0.00285251,0.00621963,0.00247439,0.0235929,0.0224142,-0.00807837],[-0.0023271,0.0469143,0.00431142,-0.0204402,-0.0223745,-0.00774278,-0.00847539,-0.00394251,0.0389318],[0.00151986,0.00431142,0.0191465,0.00673909,0.00559057,0.00676991,-0.000434418,-0.00579638,0.0155749],[0.00285251,-0.0204402,0.00673909,0.0371974,0.0156282,0.0146385,0.010273,0.00438825,0.0106748],[0.00621963,-0.0223745,0.00559057,0.0156282,0.0258642,0.0112879,0.0218765,0.0109824,-0.0062276],[0.00247439,-0.00774278,0.00676991,0.0146385,0.0112879,0.0385363,0.00725802,0.00507749,0.0131128],[0.0235929,-0.00847539,-0.000434418,0.010273,0.0218765,0.00725802,0.110753,0.0555677,-0.0178209],[0.0224142,-0.00394251,-0.00579638,0.00438825,0.0109824,0.00507749,0.0555677,0.0878987,-0.0248326],[-0.00807837,0.0389318,0.0155749,0.0106748,-0.0062276,0.0131128,-0.0178209,-0.0248326,0.0934272]]
vc = bioMatrix(9,names,values)
BIOGEME_OBJECT.VARCOVAR = vc



# The choice model is a nested logit
prob_pt = nested(V,av,nests,0)
prob_car = nested(V,av,nests,1)
prob_sm = nested(V,av,nests,2)

# Defines an itertor on the data
rowIterator('obsIter')

#Statistics
nullLoglikelihood(av,'obsIter')
choiceSet = [0,1,2]
cteLoglikelihood(choiceSet,Choice,'obsIter')
availabilityStatistics(av,'obsIter')

# Each weight is normalized so that the sum of weights is equal to the
# number of entries (1906).  
# The normalization factor has been calculated during estimation
theWeight = Weight * 1906 / 0.814484


BIOGEME_OBJECT.STATISTICS['Gender: males'] = \
                    Sum(male,'obsIter')
BIOGEME_OBJECT.STATISTICS['Gender: females'] = \
                    Sum(female,'obsIter')
BIOGEME_OBJECT.STATISTICS['Gender: unreported'] = \
                    Sum(unreportedGender,'obsIter')
BIOGEME_OBJECT.STATISTICS['Occupation: full time'] = \
                    Sum(fulltime,'obsIter')
BIOGEME_OBJECT.STATISTICS['Sum of weights'] = \
                    Sum(Weight,'obsIter')
BIOGEME_OBJECT.STATISTICS['Number of entries'] = \
                    Sum(1-exclude,'obsIter')
BIOGEME_OBJECT.STATISTICS['Normalization for elasticities PT'] = \
                    Sum(theWeight * prob_pt ,'obsIter')
BIOGEME_OBJECT.STATISTICS['Normalization for elasticities CAR'] = \
                    Sum(theWeight * prob_car ,'obsIter')
BIOGEME_OBJECT.STATISTICS['Normalization for elasticities SM'] = \
                    Sum(theWeight * prob_sm ,'obsIter')

# Define the dictionary for the simulation.
simulate = {'Prob. car': prob_car,
            'Prob. public transportation': prob_pt,
            'Prob. slow modes':prob_sm,
            'Revenue public transportation':
                   prob_pt * MarginalCostPT}

BIOGEME_OBJECT.WEIGHT = theWeight
BIOGEME_OBJECT.SIMULATE = Enumerate(simulate,'obsIter')
