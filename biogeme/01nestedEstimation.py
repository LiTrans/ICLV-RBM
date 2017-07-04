## File 01nestedEstimation.py
## Simple nested logit model for the Optima case study
## Wed May 10 10:55:12 2017

from biogeme import *
from headers import *
from loglikelihood import *
from statistics import *
from nested import *

### Three alternatives:
# CAR: automobile
# PT: public transportation
# SM: slow mode (walking, biking)

### List of parameters to be estimated
ASC_CAR = Beta('ASC_CAR',0,-10000,10000,0)
ASC_SM = Beta('ASC_SM',0,-10000,10000,0)
BETA_TIME_FULLTIME = Beta('BETA_TIME_FULLTIME',0,-10000,10000,0)
BETA_TIME_OTHER = Beta('BETA_TIME_OTHER',0,-10000,10000,0)
BETA_DIST_MALE = Beta('BETA_DIST_MALE',0,-10000,10000,0)
BETA_DIST_FEMALE = Beta('BETA_DIST_FEMALE',0,-10000,10000,0)
BETA_DIST_UNREPORTED = Beta('BETA_DIST_UNREPORTED',0,-10000,10000,0)
BETA_COST = Beta('BETA_COST',0,-10000,10000,0)


###Definition of variables:
# For numerical reasons, it is good practice to scale the data to
# that the values of the parameters are around 1.0.

# The following statements are designed to preprocess the data.
# It is like creating a new columns in the data file. This
# should be preferred to the statement like
# TimePT_scaled = Time_PT / 200.0
# which will cause the division to be reevaluated again and again,
# throuh the iterations. For models taking a long time to
# estimate, it may make a significant difference.

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
unreportedGender = DefineVariable('unreportedGender',Gender == -1)

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

NEST_NOCAR = Beta('NEST_NOCAR',1,1.0,10,0)

CAR = 1.0 , [ 1]
NO_CAR = NEST_NOCAR , [ 0, 2]
nests = CAR, NO_CAR

# All observations verifying the following expression will not be
# considered for estimation
BIOGEME_OBJECT.EXCLUDE = Choice == -1


# The choice model is a nested logit, with availability conditions
logprob = lognested(V,av,nests,Choice)

# Defines an itertor on the data
rowIterator('obsIter')

#Statistics
nullLoglikelihood(av,'obsIter')
choiceSet = [0,1,2]
cteLoglikelihood(choiceSet,Choice,'obsIter')
availabilityStatistics(av,'obsIter')

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

# Define the likelihood function for the estimation
BIOGEME_OBJECT.ESTIMATE = Sum(logprob,'obsIter')
BIOGEME_OBJECT.PARAMETERS['optimizationAlgorithm'] = "CFSQP"

