
###IMPORT NECESSARY MODULES TO RUN BIOGEME
from biogeme import *
from headers import *
from loglikelihood import *
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
coef_intercept = Beta('coef_intercept',0.0,-1000,1000,0)
coef_age_65_more = Beta('coef_age_65_more',0.0,-1000,1000,0)
coef_age_unknown = Beta('coef_age_unknown',0.0,-1000,1000,0)
coef_haveGA = Beta('coef_haveGA',0.0,-1000,1000,0)
coef_ContIncome_0_4000 = \
 Beta('coef_ContIncome_0_4000',0.0,-1000,1000,0)
coef_ContIncome_4000_6000 = \
 Beta('coef_ContIncome_4000_6000',0.0,-1000,1000,0)
coef_ContIncome_6000_8000 = \
 Beta('coef_ContIncome_6000_8000',0.0,-1000,1000,0)
coef_ContIncome_8000_10000 = \
 Beta('coef_ContIncome_8000_10000',0.0,-1000,1000,0)
coef_ContIncome_10000_more = \
 Beta('coef_ContIncome_10000_more',0.0,-1000,1000,0)
coef_moreThanOneCar = \
 Beta('coef_moreThanOneCar',0.0,-1000,1000,0)
coef_moreThanOneBike = \
 Beta('coef_moreThanOneBike',0.0,-1000,1000,0)
coef_individualHouse = \
 Beta('coef_individualHouse',0.0,-1000,1000,0)
coef_male = Beta('coef_male',0.0,-1000,1000,0)
coef_haveChildren = Beta('coef_haveChildren',0.0,-1000,1000,0)
coef_highEducation = Beta('coef_highEducation',0.0,-1000,1000,0)

### Latent variable: structural equation

# Note that the expression must be on a single line. In order to 
# write it across several lines, each line must terminate with 
# the \ symbol

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
coef_highEducation * highEducation

sigma_s = Beta('sigma_s',1,-10000,10000,1)

### Measurement equations

INTER_Envir01 = Beta('INTER_Envir01',0,-10000,10000,1)
INTER_Envir02 = Beta('INTER_Envir02',0,-10000,10000,0)
INTER_Envir03 = Beta('INTER_Envir03',0,-10000,10000,0)
INTER_Mobil11 = Beta('INTER_Mobil11',0,-10000,10000,0)
INTER_Mobil14 = Beta('INTER_Mobil14',0,-10000,10000,0)
INTER_Mobil16 = Beta('INTER_Mobil16',0,-10000,10000,0)
INTER_Mobil17 = Beta('INTER_Mobil17',0,-10000,10000,0)

B_Envir01_F1 = Beta('B_Envir01_F1',-1,-10000,10000,1)
B_Envir02_F1 = Beta('B_Envir02_F1',-1,-10000,10000,0)
B_Envir03_F1 = Beta('B_Envir03_F1',1,-10000,10000,0)
B_Mobil11_F1 = Beta('B_Mobil11_F1',1,-10000,10000,0)
B_Mobil14_F1 = Beta('B_Mobil14_F1',1,-10000,10000,0)
B_Mobil16_F1 = Beta('B_Mobil16_F1',1,-10000,10000,0)
B_Mobil17_F1 = Beta('B_Mobil17_F1',1,-10000,10000,0)



MODEL_Envir01 = INTER_Envir01 + B_Envir01_F1 * CARLOVERS
MODEL_Envir02 = INTER_Envir02 + B_Envir02_F1 * CARLOVERS
MODEL_Envir03 = INTER_Envir03 + B_Envir03_F1 * CARLOVERS
MODEL_Mobil11 = INTER_Mobil11 + B_Mobil11_F1 * CARLOVERS
MODEL_Mobil14 = INTER_Mobil14 + B_Mobil14_F1 * CARLOVERS
MODEL_Mobil16 = INTER_Mobil16 + B_Mobil16_F1 * CARLOVERS
MODEL_Mobil17 = INTER_Mobil17 + B_Mobil17_F1 * CARLOVERS

SIGMA_STAR_Envir01 = Beta('SIGMA_STAR_Envir01',10,-10000,10000,0)
SIGMA_STAR_Envir02 = Beta('SIGMA_STAR_Envir02',10,-10000,10000,0)
SIGMA_STAR_Envir03 = Beta('SIGMA_STAR_Envir03',10,-10000,10000,0)
SIGMA_STAR_Mobil11 = Beta('SIGMA_STAR_Mobil11',10,-10000,10000,0)
SIGMA_STAR_Mobil14 = Beta('SIGMA_STAR_Mobil14',10,-10000,10000,0)
SIGMA_STAR_Mobil16 = Beta('SIGMA_STAR_Mobil16',10,-10000,10000,0)
SIGMA_STAR_Mobil17 = Beta('SIGMA_STAR_Mobil17',10,-10000,10000,0)


F = {}
F['Envir01'] = Elem({0:0, \
 1:loglikelihoodregression(Envir01,MODEL_Envir01,SIGMA_STAR_Envir01)},\
  (Envir01 > 0)*(Envir01 < 6))
F['Envir02'] = Elem({0:0, \
 1:loglikelihoodregression(Envir02,MODEL_Envir02,SIGMA_STAR_Envir02)},\
  (Envir02 > 0)*(Envir02 < 6))
F['Envir03'] = Elem({0:0, \
 1:loglikelihoodregression(Envir03,MODEL_Envir03,SIGMA_STAR_Envir03)},\
  (Envir03 > 0)*(Envir03 < 6))
F['Mobil11'] = Elem({0:0, \
 1:loglikelihoodregression(Mobil11,MODEL_Mobil11,SIGMA_STAR_Mobil11)},\
  (Mobil11 > 0)*(Mobil11 < 6))
F['Mobil14'] = Elem({0:0, \
 1:loglikelihoodregression(Mobil14,MODEL_Mobil14,SIGMA_STAR_Mobil14)},\
  (Mobil14 > 0)*(Mobil14 < 6))
F['Mobil16'] = Elem({0:0, \
 1:loglikelihoodregression(Mobil16,MODEL_Mobil16,SIGMA_STAR_Mobil16)},\
  (Mobil16 > 0)*(Mobil16 < 6))
F['Mobil17'] = Elem({0:0, \
 1:loglikelihoodregression(Mobil17,MODEL_Mobil17,SIGMA_STAR_Mobil17)},\
  (Mobil17 > 0)*(Mobil17 < 6))

loglike = bioMultSum(F)


BIOGEME_OBJECT.EXCLUDE =  (Choice   ==  -1 )



# Defines an iterator on the data
rowIterator('obsIter') 

BIOGEME_OBJECT.ESTIMATE = Sum(loglike,'obsIter')
BIOGEME_OBJECT.PARAMETERS['optimizationAlgorithm'] = "CFSQP"

