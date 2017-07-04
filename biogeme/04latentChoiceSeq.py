
###IMPORT NECESSARY MODULES TO RUN BIOGEME
from biogeme import *
from headers import *
from loglikelihood import *
from distributions import *
from statistics import *

### Variables

# Piecewise linear definition of income
ScaledIncome = DefineVariable('ScaledIncome',\
                  CalculatedIncome / 1000)
ContIncome_0_4000 = DefineVariable('ContIncome_0_4000',\
                  min(ScaledIncome,4))
ContIncome_4000_6000 = DefineVariable('ContIncome_4000_6000',\
                  max(0,min(ScaledIncome-4,2)))
ContIncome_6000_8000 = DefineVariable('ContIncome_6000_8000',\
                  max(0,min(ScaledIncome-6,2)))
ContIncome_8000_10000 = DefineVariable('ContIncome_8000_10000',\
                  max(0,min(ScaledIncome-8,2)))
ContIncome_10000_more = DefineVariable('ContIncome_10000_more',\
                  max(0,ScaledIncome-10))


age_65_more = DefineVariable('age_65_more',age >= 65)
moreThanOneCar = DefineVariable('moreThanOneCar',NbCar > 1)
moreThanOneBike = DefineVariable('moreThanOneBike',NbBicy > 1)
individualHouse = DefineVariable('individualHouse',\
                                 HouseType == 1)
male = DefineVariable('male',Gender == 1)
haveChildren = DefineVariable('haveChildren',\
      ((FamilSitu == 3)+(FamilSitu == 4)) > 0)
haveGA = DefineVariable('haveGA',GenAbST == 1) 
highEducation = DefineVariable('highEducation', Education >= 6)

### Coefficients
coef_intercept = Beta('coef_intercept',0.398165,-1000,1000,1 )
coef_age_65_more = Beta('coef_age_65_more',0.0716533,-1000,1000,1 )
coef_haveGA = Beta('coef_haveGA',-0.578005,-1000,1000,1 )
coef_ContIncome_0_4000 = \
 Beta('coef_ContIncome_0_4000',0.0902761,-1000,1000,1 )
coef_ContIncome_4000_6000 = \
 Beta('coef_ContIncome_4000_6000',-0.221283,-1000,1000,1 )
coef_ContIncome_6000_8000 = \
 Beta('coef_ContIncome_6000_8000',0.259466,-1000,1000,1 )
coef_ContIncome_8000_10000 = \
 Beta('coef_ContIncome_8000_10000',-0.523049,-1000,1000,1 )
coef_ContIncome_10000_more = \
 Beta('coef_ContIncome_10000_more',0.084351,-1000,1000,1 )
coef_moreThanOneCar = \
 Beta('coef_moreThanOneCar',0.53301,-1000,1000,1 )
coef_moreThanOneBike = \
 Beta('coef_moreThanOneBike',-0.277122,-1000,1000,1 )
coef_individualHouse = \
 Beta('coef_individualHouse',-0.0885649,-1000,1000,1 )
coef_male = Beta('coef_male',0.0663476,-1000,1000,1 )
coef_haveChildren = Beta('coef_haveChildren',-0.0376042,-1000,1000,1 )
coef_highEducation = Beta('coef_highEducation',-0.246687,-1000,1000,1 )

### Latent variable: structural equation

# Note that the expression must be on a single line. In order to 
# write it across several lines, each line must terminate with 
# the \ symbol

omega = RandomVariable('omega')
density = normalpdf(omega) 
sigma_s = Beta('sigma_s',1,-1000,1000,1)

CARLOVERS = \
coef_intercept +\
coef_age_65_more * age_65_more +\
coef_ContIncome_0_4000 * ContIncome_0_4000 +\
coef_ContIncome_4000_6000 * ContIncome_4000_6000 +\
coef_ContIncome_6000_8000 * ContIncome_6000_8000 +\
coef_ContIncome_8000_10000 * ContIncome_8000_10000 +\
coef_ContIncome_10000_more * ContIncome_10000_more +\
coef_moreThanOneCar * moreThanOneCar +\
coef_moreThanOneBike * moreThanOneBike +\
coef_individualHouse * individualHouse +\
coef_male * male +\
coef_haveChildren * haveChildren +\
coef_haveGA * haveGA +\
coef_highEducation * highEducation +\
sigma_s * omega


# Choice model


ASC_CAR	 = Beta('ASC_CAR',0,-10000,10000,0)
ASC_PT	 = Beta('ASC_PT',0,-10000,10000,1)
ASC_SM	 = Beta('ASC_SM',0,-10000,10000,0)
BETA_COST_HWH = Beta('BETA_COST_HWH',0.0,-10000,10000,0 )
BETA_COST_OTHER = Beta('BETA_COST_OTHER',0.0,-10000,10000,0 )
BETA_DIST	 = Beta('BETA_DIST',0.0,-10000,10000,0)
BETA_TIME_CAR_REF = Beta('BETA_TIME_CAR_REF',0.0,-10000,0,0)
BETA_TIME_CAR_CL = Beta('BETA_TIME_CAR_CL',0.0,-10,10,0)
BETA_TIME_PT_REF = Beta('BETA_TIME_PT_REF',0.0,-10000,0,0 )
BETA_TIME_PT_CL = Beta('BETA_TIME_PT_CL',0.0,-10,10,0)
BETA_WAITING_TIME = Beta('BETA_WAITING_TIME',0.0,-10000,10000,0 )

TimePT_scaled  = DefineVariable('TimePT_scaled', TimePT   /  200 )
TimeCar_scaled  = DefineVariable('TimeCar_scaled', TimeCar   /  200 )
MarginalCostPT_scaled  = \
 DefineVariable('MarginalCostPT_scaled', MarginalCostPT   /  10 )
CostCarCHF_scaled  = \
 DefineVariable('CostCarCHF_scaled', CostCarCHF   /  10 )
distance_km_scaled  = \
 DefineVariable('distance_km_scaled', distance_km   /  5 )
PurpHWH = DefineVariable('PurpHWH', TripPurpose == 1)
PurpOther = DefineVariable('PurpOther', TripPurpose != 1)

### DEFINITION OF UTILITY FUNCTIONS:

BETA_TIME_PT = BETA_TIME_PT_REF * exp(BETA_TIME_PT_CL * CARLOVERS)

V0 = ASC_PT + \
     BETA_TIME_PT * TimePT_scaled + \
     BETA_WAITING_TIME * WaitingTimePT + \
     BETA_COST_HWH * MarginalCostPT_scaled * PurpHWH  +\
     BETA_COST_OTHER * MarginalCostPT_scaled * PurpOther

BETA_TIME_CAR = BETA_TIME_CAR_REF * exp(BETA_TIME_CAR_CL * CARLOVERS)

V1 = ASC_CAR + \
      BETA_TIME_CAR * TimeCar_scaled + \
      BETA_COST_HWH * CostCarCHF_scaled * PurpHWH  + \
      BETA_COST_OTHER * CostCarCHF_scaled * PurpOther 

V2 = ASC_SM + BETA_DIST * distance_km_scaled

# Associate utility functions with the numbering of alternatives
V = {0: V0,
     1: V1,
     2: V2}

# Associate the availability conditions with the alternatives.
# In this example all alternatives are available for each individual.
av = {0: 1,
      1: 1,
      2: 1}

# The choice model is a logit, conditional to the value of the latent variable
condprob = bioLogit(V,av,Choice)

prob = Integrate(condprob * density,'omega')

BIOGEME_OBJECT.EXCLUDE =  (Choice   ==  -1 )



# Defines an iterator on the data
rowIterator('obsIter') 

BIOGEME_OBJECT.ESTIMATE = Sum(log(prob),'obsIter')
BIOGEME_OBJECT.PARAMETERS['optimizationAlgorithm'] = "CFSQP"

