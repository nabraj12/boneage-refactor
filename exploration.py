import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pickle

import common as cm
logger=cm.log_handler()

def df_exploration(train_csv, valid_csv, test_csv):
	""" Explorer data and create image file names:
	Inputs- train, valid and test csv files
	Outputs- train, valid, and test Dataframes"""

	#Check if file exits
	if not os.path.exists(train_csv):
		logger.error("path {} does not exit".format(train_csv))

	if not os.path.exists(valid_csv):
		logger.error("path {} does not exit".format(valid_csv))
	if not os.path.exists(test_csv):
		logger.error("path {} does not exit".format(test_csv))
	
	# loading dataframes
	train_df = pd.read_csv(train_csv)
	valid_df = pd.read_csv(valid_csv)
	test_df = pd.read_csv(test_csv)

	# Creating image file into a Dataframe
	train_df['id'] = train_df['id'].apply(lambda x: str(x) + '.png')
	valid_df['Image ID'] = valid_df['Image ID'].apply(lambda x:
	 													str(x)+'.png')
	test_df['Case ID'] = test_df['Case ID'].apply(lambda x:
														str(x)+'.png')

	# Hot encoding of Gender
	train_df['Gender'] = train_df['male'].apply(lambda x: 0 if x else 1)
	valid_df['Gender'] = valid_df['male'].apply(lambda x: 0 if x else 1)
	test_df['Gender'] = test_df['Sex'].apply(lambda x: 0 if x=='M' else 1)
	test_df.rename(columns={'Ground truth bone age (months)':
								 'Age(months)'}, inplace=True)
	# Z-core of the label
	mean_age = train_df.boneage.mean()
	std_age = train_df.boneage.std()
	train_df['scale_bone_age_z'] = (train_df['boneage'] - 
												mean_age)/(std_age)
	valid_df['scale_bone_age_z'] = (valid_df['Bone Age (months)']-
												mean_age) / (std_age)
	with open('mean_std_age.pkl', 'wb') as f:
		pickle.dump([mean_age, std_age], f)

	return [train_df, valid_df, test_df] 

def my_plots(df):

	""" Gender Count, age histogram, Age distribution with each gender:
	Inputs- datafram with Column Names Gender, scale_bone_age, boneage
	Outputs- None. Displays plots"""

	plt.figure(), sns.countplot(x=df['Gender'])
	plt.title('Male(0) and Female(1) count')
	# Plotting a histogram for train bone ages (z-score)
	plt.figure(), df['scale_bone_age_z'].hist(color='red')
	plt.xlabel('bone age z score')
	plt.ylabel('# of children')
	plt.title('# of children vs bone age z score-train images')

	# Distribution of age within each gender
	male = df[df['Gender'] == 1]
	female = df[df['Gender'] == 0]

	fig, ax = plt.subplots(2, 1)
	ax[0].hist(male['boneage'], color='blue')
	ax[0].set_ylabel('# of boys')
	ax[1].hist(female['boneage'], color='red')
	ax[1].set_xlabel('Age in months')
	ax[1].set_ylabel('# of girls')
	fig.set_size_inches((10, 7))
	fig.suptitle("Male and Female age separate histogram")
	plt.show()

	return None

if __name__ == "__main__":
    
	# CSV file path + name          
	csv_train = os.path.join(os.path.dirname(__file__), 
										'mini_dataset/train.csv')
	csv_valid = os.path.join(os.path.dirname(__file__), 
										'mini_dataset/validation.csv')
	csv_test = os.path.join(os.path.dirname(__file__),
										 'mini_dataset/test.csv')

	df_list = df_exploration(csv_train, csv_valid, csv_test)

	train_df, valid_df, test_df = df_list

	my_plots(train_df)

	print(train_df.head())
	print(valid_df.head())
	print(test_df.head())
    