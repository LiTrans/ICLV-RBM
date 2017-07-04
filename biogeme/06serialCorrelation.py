
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
coef_intercept = Beta('coef_intercept',0.37254,-1000,1000,0,'coef_intercept' )
coef_age_65_more = Beta('coef_age_65_more',0.0403268,-1000,1000,0,'coef_age_65_more' )
coef_haveGA = Beta('coef_haveGA',-0.744873,-1000,1000,0,'coef_haveGA' )
coef_ContIncome_0_4000 = Beta('coef_ContIncome_0_4000',0.145815,-1000,1000,0,'coef_ContIncome_0_4000' )
coef_ContIncome_4000_6000 = Beta('coef_ContIncome_4000_6000',-0.279398,-1000,1000,0,'coef_ContIncome_4000_6000' )
coef_ContIncome_6000_8000 = Beta('coef_ContIncome_6000_8000',0.320907,-1000,1000,0,'coef_ContIncome_6000_8000' )
coef_ContIncome_8000_10000 = Beta('coef_ContIncome_8000_10000',-0.666234,-1000,1000,0,'coef_ContIncome_8000_10000' )
coef_ContIncome_10000_more = Beta('coef_ContIncome_10000_more',0.11851,-1000,1000,0,'coef_ContIncome_10000_more' )
coef_moreThanOneCar = Beta('coef_moreThanOneCar',0.710596,-1000,1000,0,'coef_moreThanOneCar' )
coef_moreThanOneBike = Beta('coef_moreThanOneBike',-0.365074,-1000,1000,0,'coef_moreThanOneBike' )
coef_individualHouse = Beta('coef_individualHouse',-0.116479,-1000,1000,0,'coef_individualHouse' )
coef_male = Beta('coef_male',0.0775805,-1000,1000,0,'coef_male' )
coef_haveChildren = Beta('coef_haveChildren',-0.0275616,-1000,1000,0,'coef_haveChildren' )
coef_highEducation = Beta('coef_highEducation',-0.265516,-1000,1000,0,'coef_highEducation' )

### Latent variable: structural equation

# Note that the expression must be on a single line. In order to 
# write it across several lines, each line must terminate with 
# the \ symbol

omega = bioDraws('omega')
sigma_s = Beta('sigma_s',0.855306,-1000,1000,0,'sigma_s' )

#
# Deal with serial correlation by including an error component that is individual specific
#
errorComponent = bioDraws('errorComponent')
ec_sigma = Beta('ec_sigma',1,-1000,1000,0)

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
sigma_s * omega+\
ec_sigma * errorComponent


### Measurement equations

INTER_Envir01 = Beta('INTER_Envir01',0,-10000,10000,1)
INTER_Envir02 = Beta('INTER_Envir02',0.459881,-10000,10000,0,'INTER_Envir02' )
INTER_Envir03 = Beta('INTER_Envir03',-0.366801,-10000,10000,0,'INTER_Envir03' )
INTER_Mobil11 = Beta('INTER_Mobil11',0.418153,-10000,10000,0,'INTER_Mobil11' )
INTER_Mobil14 = Beta('INTER_Mobil14',-0.172704,-10000,10000,0,'INTER_Mobil14' )
INTER_Mobil16 = Beta('INTER_Mobil16',0.147506,-10000,10000,0,'INTER_Mobil16' )
INTER_Mobil17 = Beta('INTER_Mobil17',0.139642,-10000,10000,0,'INTER_Mobil17' )

B_Envir01_F1 = Beta('B_Envir01_F1',-1,-10000,10000,1)
B_Envir02_F1 = Beta('B_Envir02_F1',-0.458776,-10000,10000,0,'B_Envir02_F1' )
B_Envir03_F1 = Beta('B_Envir03_F1',0.484092,-10000,10000,0,'B_Envir03_F1' )
B_Mobil11_F1 = Beta('B_Mobil11_F1',0.571806,-10000,10000,0,'B_Mobil11_F1' )
B_Mobil14_F1 = Beta('B_Mobil14_F1',0.575274,-10000,10000,0,'B_Mobil14_F1' )
B_Mobil16_F1 = Beta('B_Mobil16_F1',0.524587,-10000,10000,0,'B_Mobil16_F1' )
B_Mobil17_F1 = Beta('B_Mobil17_F1',0.514145,-10000,10000,0,'B_Mobil17_F1' )



MODEL_Envir01 = INTER_Envir01 + B_Envir01_F1 * CARLOVERS
MODEL_Envir02 = INTER_Envir02 + B_Envir02_F1 * CARLOVERS
MODEL_Envir03 = INTER_Envir03 + B_Envir03_F1 * CARLOVERS
MODEL_Mobil11 = INTER_Mobil11 + B_Mobil11_F1 * CARLOVERS
MODEL_Mobil14 = INTER_Mobil14 + B_Mobil14_F1 * CARLOVERS
MODEL_Mobil16 = INTER_Mobil16 + B_Mobil16_F1 * CARLOVERS
MODEL_Mobil17 = INTER_Mobil17 + B_Mobil17_F1 * CARLOVERS

SIGMA_STAR_Envir01 = Beta('SIGMA_STAR_Envir01',1,-10000,10000,1,'SIGMA_STAR_Envir01' )
SIGMA_STAR_Envir02 = Beta('SIGMA_STAR_Envir02',0.91756,-10000,10000,0,'SIGMA_STAR_Envir02' )
SIGMA_STAR_Envir03 = Beta('SIGMA_STAR_Envir03',0.856537,-10000,10000,0,'SIGMA_STAR_Envir03' )
SIGMA_STAR_Mobil11 = Beta('SIGMA_STAR_Mobil11',0.894838,-10000,10000,0,'SIGMA_STAR_Mobil11' )
SIGMA_STAR_Mobil14 = Beta('SIGMA_STAR_Mobil14',0.759384,-10000,10000,0,'SIGMA_STAR_Mobil14' )
SIGMA_STAR_Mobil16 = Beta('SIGMA_STAR_Mobil16',0.873045,-10000,10000,0,'SIGMA_STAR_Mobil16' )
SIGMA_STAR_Mobil17 = Beta('SIGMA_STAR_Mobil17',0.876418,-10000,10000,0,'SIGMA_STAR_Mobil17' )

delta_1 = Beta('delta_1',0.327742,0,10,0 )
delta_2 = Beta('delta_2',0.989242,0,10,0 )
tau_1 = -delta_1 - delta_2
tau_2 = -delta_1 
tau_3 = delta_1
tau_4 = delta_1 + delta_2

Envir01_tau_1 = (tau_1-MODEL_Envir01) / SIGMA_STAR_Envir01
Envir01_tau_2 = (tau_2-MODEL_Envir01) / SIGMA_STAR_Envir01
Envir01_tau_3 = (tau_3-MODEL_Envir01) / SIGMA_STAR_Envir01
Envir01_tau_4 = (tau_4-MODEL_Envir01) / SIGMA_STAR_Envir01
IndEnvir01 = {
    1: bioNormalCdf(Envir01_tau_1),
    2: bioNormalCdf(Envir01_tau_2)-bioNormalCdf(Envir01_tau_1),
    3: bioNormalCdf(Envir01_tau_3)-bioNormalCdf(Envir01_tau_2),
    4: bioNormalCdf(Envir01_tau_4)-bioNormalCdf(Envir01_tau_3),
    5: 1-bioNormalCdf(Envir01_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0
}

P_Envir01 = Elem(IndEnvir01, Envir01)


Envir02_tau_1 = (tau_1-MODEL_Envir02) / SIGMA_STAR_Envir02
Envir02_tau_2 = (tau_2-MODEL_Envir02) / SIGMA_STAR_Envir02
Envir02_tau_3 = (tau_3-MODEL_Envir02) / SIGMA_STAR_Envir02
Envir02_tau_4 = (tau_4-MODEL_Envir02) / SIGMA_STAR_Envir02
IndEnvir02 = {
    1: bioNormalCdf(Envir02_tau_1),
    2: bioNormalCdf(Envir02_tau_2)-bioNormalCdf(Envir02_tau_1),
    3: bioNormalCdf(Envir02_tau_3)-bioNormalCdf(Envir02_tau_2),
    4: bioNormalCdf(Envir02_tau_4)-bioNormalCdf(Envir02_tau_3),
    5: 1-bioNormalCdf(Envir02_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0
}

P_Envir02 = Elem(IndEnvir02, Envir02)

Envir03_tau_1 = (tau_1-MODEL_Envir03) / SIGMA_STAR_Envir03
Envir03_tau_2 = (tau_2-MODEL_Envir03) / SIGMA_STAR_Envir03
Envir03_tau_3 = (tau_3-MODEL_Envir03) / SIGMA_STAR_Envir03
Envir03_tau_4 = (tau_4-MODEL_Envir03) / SIGMA_STAR_Envir03
IndEnvir03 = {
    1: bioNormalCdf(Envir03_tau_1),
    2: bioNormalCdf(Envir03_tau_2)-bioNormalCdf(Envir03_tau_1),
    3: bioNormalCdf(Envir03_tau_3)-bioNormalCdf(Envir03_tau_2),
    4: bioNormalCdf(Envir03_tau_4)-bioNormalCdf(Envir03_tau_3),
    5: 1-bioNormalCdf(Envir03_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0
}

P_Envir03 = Elem(IndEnvir03, Envir03)

Mobil11_tau_1 = (tau_1-MODEL_Mobil11) / SIGMA_STAR_Mobil11
Mobil11_tau_2 = (tau_2-MODEL_Mobil11) / SIGMA_STAR_Mobil11
Mobil11_tau_3 = (tau_3-MODEL_Mobil11) / SIGMA_STAR_Mobil11
Mobil11_tau_4 = (tau_4-MODEL_Mobil11) / SIGMA_STAR_Mobil11
IndMobil11 = {
    1: bioNormalCdf(Mobil11_tau_1),
    2: bioNormalCdf(Mobil11_tau_2)-bioNormalCdf(Mobil11_tau_1),
    3: bioNormalCdf(Mobil11_tau_3)-bioNormalCdf(Mobil11_tau_2),
    4: bioNormalCdf(Mobil11_tau_4)-bioNormalCdf(Mobil11_tau_3),
    5: 1-bioNormalCdf(Mobil11_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0
}

P_Mobil11 = Elem(IndMobil11, Mobil11)

Mobil14_tau_1 = (tau_1-MODEL_Mobil14) / SIGMA_STAR_Mobil14
Mobil14_tau_2 = (tau_2-MODEL_Mobil14) / SIGMA_STAR_Mobil14
Mobil14_tau_3 = (tau_3-MODEL_Mobil14) / SIGMA_STAR_Mobil14
Mobil14_tau_4 = (tau_4-MODEL_Mobil14) / SIGMA_STAR_Mobil14
IndMobil14 = {
    1: bioNormalCdf(Mobil14_tau_1),
    2: bioNormalCdf(Mobil14_tau_2)-bioNormalCdf(Mobil14_tau_1),
    3: bioNormalCdf(Mobil14_tau_3)-bioNormalCdf(Mobil14_tau_2),
    4: bioNormalCdf(Mobil14_tau_4)-bioNormalCdf(Mobil14_tau_3),
    5: 1-bioNormalCdf(Mobil14_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0
}

P_Mobil14 = Elem(IndMobil14, Mobil14)

Mobil16_tau_1 = (tau_1-MODEL_Mobil16) / SIGMA_STAR_Mobil16
Mobil16_tau_2 = (tau_2-MODEL_Mobil16) / SIGMA_STAR_Mobil16
Mobil16_tau_3 = (tau_3-MODEL_Mobil16) / SIGMA_STAR_Mobil16
Mobil16_tau_4 = (tau_4-MODEL_Mobil16) / SIGMA_STAR_Mobil16
IndMobil16 = {
    1: bioNormalCdf(Mobil16_tau_1),
    2: bioNormalCdf(Mobil16_tau_2)-bioNormalCdf(Mobil16_tau_1),
    3: bioNormalCdf(Mobil16_tau_3)-bioNormalCdf(Mobil16_tau_2),
    4: bioNormalCdf(Mobil16_tau_4)-bioNormalCdf(Mobil16_tau_3),
    5: 1-bioNormalCdf(Mobil16_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0
}

P_Mobil16 = Elem(IndMobil16, Mobil16)

Mobil17_tau_1 = (tau_1-MODEL_Mobil17) / SIGMA_STAR_Mobil17
Mobil17_tau_2 = (tau_2-MODEL_Mobil17) / SIGMA_STAR_Mobil17
Mobil17_tau_3 = (tau_3-MODEL_Mobil17) / SIGMA_STAR_Mobil17
Mobil17_tau_4 = (tau_4-MODEL_Mobil17) / SIGMA_STAR_Mobil17
IndMobil17 = {
    1: bioNormalCdf(Mobil17_tau_1),
    2: bioNormalCdf(Mobil17_tau_2)-bioNormalCdf(Mobil17_tau_1),
    3: bioNormalCdf(Mobil17_tau_3)-bioNormalCdf(Mobil17_tau_2),
    4: bioNormalCdf(Mobil17_tau_4)-bioNormalCdf(Mobil17_tau_3),
    5: 1-bioNormalCdf(Mobil17_tau_4),
    6: 1.0,
    -1: 1.0,
    -2: 1.0
}

P_Mobil17 = Elem(IndMobil17, Mobil17)

# Choice model


ASC_CAR = Beta('ASC_CAR',0.703144,-10000,10000,0,'ASC_CAR' )
ASC_PT	 = Beta('ASC_PT',0,-10000,10000,1)
ASC_SM = Beta('ASC_SM',0.261217,-10000,10000,0,'ASC_SM' )
BETA_COST_HWH = Beta('BETA_COST_HWH',-1.43061,-10000,10000,0,'BETA_COST_HWH' )
BETA_COST_OTHER = Beta('BETA_COST_OTHER',-0.52555,-10000,10000,0,'BETA_COST_OTHER' )
BETA_DIST = Beta('BETA_DIST',-1.41373,-10000,10000,0,'BETA_DIST' )
BETA_TIME_CAR_REF = Beta('BETA_TIME_CAR_REF',-9.49633,-10000,0,0,'BETA_TIME_CAR_REF' )
BETA_TIME_CAR_CL = Beta('BETA_TIME_CAR_CL',-0.955607,-10,10,0,'BETA_TIME_CAR_CL' )
BETA_TIME_PT_REF = Beta('BETA_TIME_PT_REF',-3.22241,-10000,0,0,'BETA_TIME_PT_REF' )
BETA_TIME_PT_CL = Beta('BETA_TIME_PT_CL',-0.456234,-10,10,0,'BETA_TIME_PT_CL' )
BETA_WAITING_TIME = Beta('BETA_WAITING_TIME',-0.0204706,-10000,10000,0,'BETA_WAITING_TIME' )

TimePT_scaled  = DefineVariable('TimePT_scaled', TimePT   /  200 )
TimeCar_scaled  = DefineVariable('TimeCar_scaled', TimeCar   /  200 )
MarginalCostPT_scaled  = DefineVariable('MarginalCostPT_scaled', MarginalCostPT   /  10 )
CostCarCHF_scaled  = DefineVariable('CostCarCHF_scaled', CostCarCHF   /  10 )
distance_km_scaled  = DefineVariable('distance_km_scaled', distance_km   /  5 )
PurpHWH = DefineVariable('PurpHWH', TripPurpose == 1)
PurpOther = DefineVariable('PurpOther', TripPurpose != 1)


### DEFINITION OF UTILITY FUNCTIONS:

BETA_TIME_PT = BETA_TIME_PT_REF * exp(BETA_TIME_PT_CL * CARLOVERS)

V0 = ASC_PT + \
     BETA_TIME_PT * TimePT_scaled + \
     BETA_WAITING_TIME * WaitingTimePT + \
     BETA_COST_HWH * MarginalCostPT_scaled * PurpHWH  +\
     BETA_COST_OTHER * MarginalCostPT_scaled * PurpOther +\
     ec_sigma * errorComponent

BETA_TIME_CAR = BETA_TIME_CAR_REF * exp(BETA_TIME_CAR_CL * CARLOVERS)

V1 = ASC_CAR + \
      BETA_TIME_CAR * TimeCar_scaled + \
      BETA_COST_HWH * CostCarCHF_scaled * PurpHWH  + \
      BETA_COST_OTHER * CostCarCHF_scaled * PurpOther+\
      ec_sigma * errorComponent 

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

condlike = P_Envir01 * \
          P_Envir02 * \
          P_Envir03 * \
          P_Mobil11 * \
          P_Mobil14 * \
          P_Mobil16 * \
          P_Mobil17 * \
          condprob

loglike = log(MonteCarlo(condlike))


BIOGEME_OBJECT.EXCLUDE =  (Choice   ==  -1 )



# Defines an iterator on the data
rowIterator('obsIter') 

BIOGEME_OBJECT.ESTIMATE = Sum(loglike,'obsIter')

BIOGEME_OBJECT.PARAMETERS['numberOfThreads'] = "3"
BIOGEME_OBJECT.PARAMETERS['RandomDistribution'] = "MLHS"
BIOGEME_OBJECT.DRAWS = { 'omega': 'NORMAL','errorComponent': 'NORMAL' }
BIOGEME_OBJECT.PARAMETERS['NbrOfDraws'] = "500"
