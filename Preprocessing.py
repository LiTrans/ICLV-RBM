import numpy as np
import pandas as pd


class Preprocessing(object):
	""" Configures input data for estimation
		Invoke the class to prepare data structure

	functions
	---------
	data(self)
		:return: a list of dataset

	"""

	def __init__(self, csvName='US_SP_Restructured.csv'):
		"""
		Parameters
		----------
		:string csvName: Name of csv file
						 default: 'US_SP_Restructured.csv'
		"""
		# read csv
		self.df = pd.read_csv(csvName)

		# new column from index
		self.df['index'] = self.df.index

	def extractData(self):
		# extract alternative specific variables
		cost = pd.melt(
			self.df,
			id_vars=['quest', 'index'],
			value_vars=[
				'Car_Cost', 'CarRental_Cost', 'Bus_Cost', 'Plane_Cost',
				'Train_Cost', 'TrH_Cost'],
			value_name='cost')

		tt = pd.melt(
			self.df,
			id_vars=['quest', 'index'],
			value_vars=[
				'Car_TT', 'CarRental_TT', 'Bus_TT', 'Plane_TT', 'Train_TT',
				'TrH_TT'],
			value_name='tt')

		relib = pd.melt(
			self.df,
			id_vars=['quest', 'index'],
			value_vars=[
				'CarRelib', 'CarRentalRelib', 'BusRelib', 'PlaneRelib',
				'TrainRelib', 'TrHRelib'],
			value_name='reliability')

		# extract generic variables
		data_RP = self.df[[
			'quest', 'index',
			'DrvLicens', 'PblcTrst', 'Ag1825', 'Ag2545', 'Ag4565', 'Ag65M',
			'Male', 'Fulltime', 'PrtTime', 'Unemplyd',
			'Edu_Highschl', 'Edu_BSc', 'Edu_MscPhD',
			'HH_Veh0', 'HH_Veh1', 'HH_Veh2M',
			'HH_Adult1', 'HH_Adult2', 'HH_Adult3M',
			'HH_Chld0', 'HH_Chld1', 'HH_Chld2M',
			'HH_Inc020K', 'HH_Inc2060K', 'HH_Inc60KM',
			'HH_Sngl', 'HH_SnglParent', 'HH_AllAddults',
			'HH_Nuclear', 'P_Chld',
			'BusCrwd', 'CarMorning', 'CarAfternoon', 'CarEve',
			'O_MTL_US_max', 'O_Odr_US_max', 'D_Bstn_max', 'D_NYC_max', 'D_Maine_max',
			'Tp_Onewy_max', 'Tp_2way_max',
			'Tp_h06_max', 'Tp_h69_max', 'Tp_h915_max', 'Tp_h1519_max',
			'Tp_h1924_max', 'Tp_h1524_max',
			'Tp_Y2016_max', 'Tp_Y2017_max',
			'Tp_Wntr_max', 'Tp_Sprng_max', 'Tp_Sumr_max', 'Tp_Fall_max',
			'Tp_CarDrv_max', 'Tp_CarPsngr_max', 'Tp_CarShrRnt_max',
			'Tp_Train_max', 'Tp_Bus_max', 'Tp_Plane_max', 'Tp_ModOdr_max',
			'Tp_WrkSkl_max', 'Tp_Leisr_max', 'Tp_Shpng_max', 'Tp_ActOdr_max',
			'Tp_NHotel1_max', 'Tp_NHotel2_max', 'Tp_NHotel3M_max',
			'Tp_FreqMonthlMulti_max', 'Tp_FreqYearMulti_max',
			'Tp_FreqYear1_max',
			'Envrn_Car', 'Envrn_Train', 'Envrn_Bus', 'Envrn_Plane',
			'Safe_Car', 'Safe_Train', 'Safe_Bus', 'Safe_Plane',
			'Comf_Car', 'Comf_Train', 'Comf_Bus', 'Comf_Plane',
			'Import_Cost', 'Import_TT', 'Import_Relib', 'Import_StartTime',
			'Import_Freq', 'Import_Onboard', 'Import_Crwding'
		]]

		# extract alternatives
		data_choice = self.df[['quest', 'index', 'New_SP_Choice']]

		# extract availability
		data_avail = self.df[[
			'quest', 'index',
			'AV_Car', 'AV_CarRental', 'AV_Bus', 'AV_Plane', 'AV_Train',
			'AV_TrH'
		]]

		# extract indicators
		data_ind = self.df[[
			'quest', 'index',
			'Envrn_Car', 'Envrn_Train', 'Envrn_Bus', 'Envrn_Plane',
			'Safe_Car', 'Safe_Train', 'Safe_Bus', 'Safe_Plane',
			'Comf_Car', 'Comf_Train', 'Comf_Bus', 'Comf_Plane'
		]]

		data_choice = data_choice.sort_values(['quest', 'index'])
		cost = cost.sort_values(['quest', 'index', 'variable'])
		tt = tt.sort_values(['quest', 'index', 'variable'])
		relib = relib.sort_values(['quest', 'index', 'variable'])
		data_RP = data_RP.sort_values(['quest', 'index'])
		data_avail = data_avail.sort_values(['quest', 'index'])
		data_ind = data_ind.sort_values(['quest', 'index'])

		# make a copy and merge alternative specific variables
		data_SP = cost
		data_SP['tt'] = tt['tt']
		data_SP['relib'] = relib['reliability']
		data_SP['choice'] = data_SP['variable'].str.split('_', expand=True)[0]
		data_SP = data_SP.reset_index(drop=True)

		# check if everything is in order
		print(data_SP.head(6))

		# extract data arrays
		self.dataset_y = data_choice[['New_SP_Choice']]
		self.dataset_x_ng = data_SP[['cost', 'tt', 'relib']]
		self.dataset_x_g = data_RP[[
			'DrvLicens', 'PblcTrst',
			'Ag1825', 'Ag2545', 'Ag4565', 'Ag65M',
			'Male', 'Fulltime', 'PrtTime', 'Unemplyd',
			'Edu_Highschl', 'Edu_BSc', 'Edu_MscPhD',
			'HH_Veh0', 'HH_Veh1', 'HH_Veh2M',
			'HH_Adult1', 'HH_Adult2', 'HH_Adult3M',
			'HH_Chld0', 'HH_Chld1', 'HH_Chld2M',
			# 'HH_Inc020K', 'HH_Inc2060K', 'HH_Inc60KM',
			# 'HH_Sngl', 'HH_SnglParent', HH_AllAddults',
			# 'HH_Nuclear', 'P_Chld',
			# 'O_MTL_US_max', 'O_Odr_US_max',
			# 'D_Bstn_max', 'D_NYC_max', 'D_Maine_max',
			# 'Tp_Onewy_max', 'Tp_2way_max',
			# 'Tp_h06_max', 'Tp_h69_max', 'Tp_h915_max', 'Tp_h1519_max',
			# 'Tp_h1924_max', 'Tp_h1524_max',
			# 'Tp_Y2016_max', 'Tp_Y2017_max',
			# 'Tp_Wntr_max', 'Tp_Sprng_max', 'Tp_Sumr_max', 'Tp_Fall_max',
			# 'Tp_CarDrv_max', 'Tp_CarPsngr_max', 'Tp_CarShrRnt_max',
			# 'Tp_Train_max', 'Tp_Bus_max', 'Tp_Plane_max', 'Tp_ModOdr_max',
			# 'Tp_WrkSkl_max', 'Tp_Leisr_max', 'Tp_Shpng_max',
			# 'Tp_ActOdr_max',
			# 'Tp_NHotel1_max', 'Tp_NHotel2_max', 'Tp_NHotel3M_max',
			# 'Tp_FreqMonthlMulti_max', 'Tp_FreqYearMulti_max',
			# 'Tp_FreqYear1_max',
		]]
		self.dataset_avail = data_avail[[
			'AV_Bus', 'AV_CarRental', 'AV_Car', 'AV_Plane', 'AV_TrH',
			'AV_Train'
		]]
		self.dataset_ind = data_ind[[
			'Envrn_Car', 'Envrn_Train', 'Envrn_Bus', 'Envrn_Plane',
			'Safe_Car', 'Safe_Train', 'Safe_Bus', 'Safe_Plane',
			'Comf_Car', 'Comf_Train', 'Comf_Bus', 'Comf_Plane'
		]]

		n = self.df.shape[0]
		y = self.dataset_y.values.reshape(n,)
		x_ng = self.dataset_x_ng.values.reshape(n, 6, -1)/100.
		x_g = self.dataset_x_g.values
		avail = self.dataset_avail.values
		ind = self.dataset_ind.values

		return x_ng, x_g, y, avail, ind
