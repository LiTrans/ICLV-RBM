import numpy as np
import pandas as pd

def dataWrangling():
	## load US_SP_Restructed.csv ##
	df = pd.read_csv('US_SP_Restructured.csv')
	df['index'] = df.index

	# extract generic variables (Cost, TT, reliability)
	cost = pd.melt(df,
		id_vars=['quest', 'index'],
		value_vars=['Car_Cost', 'CarRental_Cost', 'Bus_Cost', 'Plane_Cost', 'Train_Cost', 'TrH_Cost'],
		var_name=None,
		value_name='cost', col_level=None
	).sort_values(['quest', 'index', 'variable'])

	tt = pd.melt(df,
		id_vars=['quest', 'index'],
		value_vars=['Car_TT', 'CarRental_TT', 'Bus_TT', 'Plane_TT', 'Train_TT', 'TrH_TT'],
		var_name=None,
		value_name='tt', col_level=None
	).sort_values(['quest', 'index', 'variable'])

	relib = pd.melt(df,
		id_vars=['quest', 'index'],
		value_vars=['CarRelib', 'RentalRelib', 'BusRelib', 'PlneRelib', 'TrainRelib', 'TrHRelib'],
		var_name=None,
		value_name='reliability',
		col_level=None
	).sort_values(['quest', 'index', 'variable'])

	# extract RP and socio-economic data
	rp_data = df[['quest', 'DrvLicens', 'PblcTrst', 'Ag1825', 'Ag2545', 'Ag4565', 'Ag65M', 'Male', 'Fulltime', 'PrtTime', 'Unemplyd', 'Edu_Highschl', 'Edu_BSc', 'Edu_MscPhD', 'HH_Veh0', 'HH_Veh1', 'HH_Veh2M', 'HH_Adult1',   \
	'HH_Adult2', 'HH_Adult3M', 'HH_Chld0', 'HH_Chld1', 'HH_Chld2M', 'HH_Inc020K', 'HH_Inc2060K', 'HH_Inc60KM', 'HH_Inc_Error', 'HH_Sngl', 'HH_SnglParent', 'HH_AllAddults', 'HH_Nuclear', 'P_Chld', 'Import_Cost', 'Import_TT', \
	'Import_Relib', 'Import_StartTime', 'Import_Freq', 'Import_Onboard', 'Import_Crwding', 'BusCrwd', 'CarMorning', 'CarAfternoon', 'CarEve', 'O_MTL_US_max', 'O_Odr_US_max', 'D_Bstn_max', 'D_NYC_max', 'D_Maine_max',              \
	'Tp_Onewy_max', 'Tp_2way_max', 'Tp_h06_max', 'Tp_h69_max', 'Tp_h915_max', 'Tp_h1519_max', 'Tp_h1924_max', 'Tp_h1524_max', 'Tp_Y2016_max', 'Tp_Y2017_max', 'Tp_Wntr_max', 'Tp_Sprng_max', 'Tp_Sumr_max', 'Tp_Fall_max',     \
	'Tp_CarDrv_max', 'Tp_CarPsngr_max', 'Tp_CarShrRnt_max', 'Tp_Train_max', 'Tp_Bus_max', 'Tp_Plane_max', 'Tp_ModOdr_max', 'Tp_WrkSkl_max', 'Tp_Leisr_max', 'Tp_Shpng_max', 'Tp_ActOdr_max', 'Tp_NHotel1_max',             \
	'Tp_NHotel2_max', 'Tp_NHotel3M_max', 'Tp_FreqMonthlMulti_max', 'Tp_FreqYearMulti_max', 'Tp_FreqYear1_max', 'Envrn_Car', 'Envrn_Train', 'Envrn_Bus', 'Envrn_Plane', 'Safe_Car', 'Safe_Train', 'Safe_Bus', 'Safe_Plane',      \
	'Comf_Car', 'Comf_Train', 'Comf_Bus', 'Comf_Plane', 'index']].sort_values(['quest', 'index'])

	choice_data = df[['quest', 'index', 'New_SP_Choice']].sort_values(['quest', 'index'])

	# extract availability
	availability = df[['quest', 'AV_Car', 'AV_CarRental', 'AV_Bus', 'AV_Plane', 'AV_Train', 'AV_TrH', 'index']].sort_values(['quest', 'index'])

	# make a copy and merge
	data = cost
	data['choice'] = data['variable'].str.split('_', expand=True)[0]
	data['tt'] = tt['tt']
	data['relib'] = relib['reliability']

	# reorder columns, reset index and change NaN -> 0
	data = data.fillna(0)
	data = data[['quest', 'index', 'choice', 'cost', 'tt', 'relib']].reset_index(drop=True)

	# check if everything is in order
	print(data.head(6))

	# reshape data to numpy array
	dataset_x_ng = data[['cost', 'tt', 'relib']].values.reshape(df.shape[0], 6, -1)/100 # shape:(n,i,m) (1788, 6, 3)
	dataset_x_g = rp_data[['DrvLicens', 'PblcTrst', 'Ag1825', 'Ag2545', 'Male',]].values.reshape(df.shape[0], -1)  # shape: (n, m) (1788, 2)
	dataset_y = choice_data[['New_SP_Choice']].values.reshape(df.shape[0],) 	 # shape:(n,) (1788,)
	dataset_availability = availability[['AV_Bus', 'AV_CarRental', 'AV_Car', 'AV_Plane', 'AV_TrH', 'AV_Train']]

	# check if everything is in order
	print(dataset_x_ng.shape, dataset_x_g.shape, dataset_y.shape)

	# return arrays
	return dataset_x_ng, dataset_x_g, dataset_y, dataset_availability

if __name__ == '__main__':
	dataWrangling()
