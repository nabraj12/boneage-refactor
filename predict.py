from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cbook import flatten
import pickle
import sys
import logging
import warnings
warnings.filterwarnings("ignore")

import common as cm
logger=cm.log_handler()

def test_dataframe(test_csv):
	"""Read test CSV and create test dataframe.
	Input- csv file path
	Output-test df """
	if not os.path.isfile(test_csv):
		logger.error("path {} does not exit".format(test_csv))
		sys.exit("---test csv file--- does not exits")

	test_df = pd.read_csv(test_csv)
	test_df['Case ID'] = test_df['Case ID'].apply(lambda x:
													 str(x)+'.png')

	test_df['Gender'] = test_df['Sex'].apply(lambda x:
												0 if x=='M' else 1)
	
	test_df.rename(columns={'Ground truth bone age (months)':
								 'Age(months)'}, inplace=True)

	return test_df

def img_gender_gen(gen_img, gen_gender):
    """Create a generator with image and gender as input.
    Input- image generator, gender generator
    Output-X, Y generators"""  
    
    while True:
        X1i = gen_img.next()
        X2i = gen_gender.next()
        yield [X1i[0], X2i[1]], X1i[1]

def input_generator(img_dir, df, X_COL, BATCH_SIZE, SEED, IMG_SIZE):
	"""Create X and Y for model prediction
	Input- img dir, df, X COL, BATCH SIZE, SEED, IMG_SIZE
	Output-X (feature), Y (label)"""
	imgDatGen = ImageDataGenerator()

	img = imgDatGen.flow_from_dataframe(
									dataframe=df,
		                            directory= img_dir,
		                            x_col= X_COL,
		                            y_col='Age(months)',                                            
		                            batch_size=BATCH_SIZE,
		                            seed=SEED,
		                            shuffle=True,
		                            class_mode='raw',
		                            target_size=(IMG_SIZE, IMG_SIZE),
		                            color_mode='rgb')

	gender = imgDatGen.flow_from_dataframe(
    							dataframe=df,
    							directory = img_dir,
                                x_col=X_COL,
                                y_col='Gender',
                                batch_size=BATCH_SIZE,
                                seed=SEED,
                                shuffle=True,
                                class_mode='raw', 
                                target_size=(IMG_SIZE, IMG_SIZE),
                               	color_mode='rgb')

	test_gen = img_gender_gen(img, gender)

	test_XX, test_YY = next(test_gen)

	return test_XX, test_YY

def predict_bone_age(model, BATCH_SIZE, test_XX, test_YY, pklfile):
	"""Predict bone age
	Input- model, BATCH_SIZE, test_XX, test_YY, pklfile
	Output-Predicted age in month"""

	predict_age = model.predict(test_XX, batch_size = BATCH_SIZE,
													verbose = True)

	mean_age, std_age = cm.mean_std_age(pklfile)
	
	predict_age_flat = predict_age.flatten()

	predict_age_month = mean_age+std_age*(predict_age_flat)

	print('Mean absolute distance of the test images: {} months'.
					format(round(np.mean(abs(test_YY-
						list(flatten(predict_age_month)))),1)))

	return predict_age_month

def plot_prediction(test_YY, predict_age_month):
	"""Display predicted age vs actual age plot.
	Input- Actual age(month), predicted age (month)
	Output-None"""

	# PLot-actual vs predicted age from test image
	fig, ax = plt.subplots(figsize = (7,7))

	plt.plot(test_YY, predict_age_month, 'ro')

	ax.plot(test_YY, predict_age_month, 'r.',
					label = 'predictions (xception)-test image')

	ax.plot(test_YY, test_YY, 'b-',
								label = 'actual-test image')

	ax.legend(loc = 'upper right')
	ax.set_xlabel('Actual Age (Months)')
	ax.set_ylabel('Predicted Age (Months)')
	plt.show()


if __name__== "__main__":

	IMG_SIZE = 256
	BATCH_SIZE = 200
	SEED = 42
	pklfile = "mean_std_age.pkl"
	MODEL_JSON = "model.json"
	MODEL_H5 = "model.h5"
	test_img_dir = os.path.join(os.path.dirname(__file__), 
								'mini_dataset/test_pre/')

	test_csv = os.path.join(os.path.dirname(__file__),
										 'mini_dataset/test.csv')
	test_df = test_dataframe(test_csv)

	model = cm.load_model(MODEL_JSON, MODEL_H5)

	test_XX, test_YY = input_generator(test_img_dir, test_df,
								'Case ID', BATCH_SIZE, SEED,IMG_SIZE)
	
	predict_age_month = predict_bone_age(model, BATCH_SIZE, test_XX,
													test_YY, pklfile)

	plot_prediction(test_YY, predict_age_month)

	print("======Task completed==================")





