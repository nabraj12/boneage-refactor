import tensorflow
from tensorflow.keras.metrics import mean_absolute_error
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense,Flatten
from tensorflow.keras.layers import Dropout, Conv2D,Input,concatenate
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras import Sequential
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cbook import flatten
import numpy as np
import sys
import logging
import warnings
warnings.filterwarnings("ignore")
#===================
import common as cm
logger=cm.log_handler()

pickle_file_name = 'mean_std_age.pkl'
mean_age, std_age = cm.mean_std_age(pickle_file_name)

def plot_it(history):
    """Plot training and validation error.
    Input- history (model training history)
    Output-None (display a plot)"""
    fig, ax = plt.subplots( figsize=(20,10))
    ax.plot(history.history['mae_in_months'])
    ax.plot(history.history['val_mae_in_months'])
    plt.title('Model Error')
    plt.ylabel('error')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    ax.grid(color='black')
    plt.show()

def mae_in_months(x_p, y_p):
	'''Calculaye mae in months
    Input- X value, Y value
    Output-mae'''	
	mae = mean_absolute_error((std_age*x_p + mean_age),
									(std_age*y_p + mean_age))
	return mae 

def img_gender_inputs(imgDatGen, img_dir, df, batch_size, seed,
												 img_size, X_COL):
    """Create flow_from_dataframe for image and gender
    Input- imgDatGen, img dir, df, batch size, seed, img_size, X_COL
    Output-image and gender generator"""
    
    img = imgDatGen.flow_from_dataframe(
								dataframe=df,
                                directory= img_dir,
                                x_col= X_COL,
                                y_col='scale_bone_age_z',                                            
                                batch_size=batch_size,
                                seed=seed,
                                shuffle=True,
                                class_mode='raw',
                                target_size=(img_size, img_size),
                                color_mode='rgb')

    gender = imgDatGen.flow_from_dataframe(
    							dataframe=df,
    							directory = img_dir,
                                x_col=X_COL,
                                y_col='Gender',
                                batch_size=batch_size,
                                seed=seed,
                                shuffle=True,
                                class_mode='raw', 
                                target_size=(img_size, img_size),
                               	color_mode='rgb')
    return img, gender

def img_gender_gen(gen_img, gen_gender):
    """Create a generator with image and gender as input.
    Input- image generator, gender generator
    Output-X, Y generators"""    
    while True:
        X1i = gen_img.next()
        X2i = gen_gender.next()
        yield [X1i[0], X2i[1]], X1i[1]

def get_input_train(img_dir, df, BATCH_SIZE, SEED, IMG_SIZE, X_COL):
	"""Prepare train input and label to feed to the model.
	Input- img dir, df, BATCH SIZE, SEED, IMG SIZE, X COL
	Output-train generator, step size train"""
	idg = ImageDataGenerator(zoom_range=0.2,
	                           fill_mode='nearest',
	                           rotation_range=25,  
	                           width_shift_range=0.25,  
	                           height_shift_range=0.25,  
	                           vertical_flip=False, 
	                           horizontal_flip=True,
	                           shear_range = 0.2,
	                           samplewise_center=False, 
	                           samplewise_std_normalization=False)

	img, gender = img_gender_inputs(idg,
									img_dir,
									df,
									BATCH_SIZE,
									SEED,
									IMG_SIZE,
									X_COL)

	train_gen = img_gender_gen(img, gender)

	STEP_SIZE_TRAIN = img.n//img.batch_size

	return train_gen, STEP_SIZE_TRAIN

def get_input_valid(img_dir, df, BATCH_SIZE, SEED, IMG_SIZE, X_COL):	
	"""Prepare validation input and label to feed to the model.
	Input- img dir, df, BATCH SIZE, SEED, IMG SIZE, X COL
	Output-valid generator, step size valid"""
	idg = ImageDataGenerator(width_shift_range=0.25,
	                             height_shift_range=0.25,
	                             horizontal_flip=True)

	img, gender = img_gender_inputs(idg,
									img_dir,
									df,
									BATCH_SIZE,
									SEED,
									IMG_SIZE,
									X_COL)

	valid_gen = img_gender_gen(img, gender)   

	STEP_SIZE_VALID = img.n//img.batch_size

	return valid_gen, STEP_SIZE_VALID

def two_inputs_model(IMG_SIZE):
	"""Creare model architecture.
	Input- image size
	Output-model (saves model.json)"""
	input_tensor = Input(shape=(IMG_SIZE,IMG_SIZE,3))
	x = Xception(include_top = False,weights = 'imagenet')(input_tensor)
	x.trainable = True
	x = GlobalMaxPooling2D()(x)
	x = Flatten()(x)
	x = Model(inputs=input_tensor, outputs=x)

	# Second input
	input_sex = Input(shape=(1,))
	y = Dense(32, activation="relu")(input_sex)
	y = Model(inputs=input_sex, outputs=y)
	# Combine the output of the two branches
	combined = concatenate([x.output, y.output])

	# FC layers and then a regression prediction
	z = Dense(1000, activation = 'relu')(combined) # changed from 10 to 64
	z = Dropout(0.3)(z)
	z = Dense(1000, activation = 'relu')(z) # added
	z = Dropout(0.3)(z)
	z = Dense(1, activation="linear")(z)
	# This model accepts the inputs of the two branches and
	# then output a single value
	model = Model(inputs=[x.input, y.input], outputs=z)

	#compile model
	model.compile(loss ='mse', optimizer= 'adam', metrics = [mae_in_months])
	
	# save model
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
	    json_file.write(model_json)

	#model summary
	model.summary()

	return model

def train_model(model, XY_train, STEP_SZ_TRAIN, XY_valid,
											 STEP_SZ_VALID, EPOCHS):
	"""Train the model.
	Input- model, XY_train, STEP_SZ_train, XY_valid, STEP_SZ_val, EPOCHS
	Output-history (saves model.h5) """

	#early stopping
	early_stopping = EarlyStopping(monitor='val_loss',
	                              min_delta=0,
	                              patience= 5,
	                              verbose=0, mode='auto')
	#model checkpoint
	mc = ModelCheckpoint('model.h5', monitor='val_loss',
									 mode='min', save_best_only=True)

	#reduce lr on plateau
	red_lr_plat = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
									patience=10, verbose=0, mode='auto',
									min_delta=0.0001, cooldown=0, min_lr=0)

	callbacks = [early_stopping,mc, red_lr_plat]

	# fit model
	history = model.fit(XY_train,
                        steps_per_epoch = STEP_SZ_TRAIN,
                        validation_data = XY_valid,
                        validation_steps = STEP_SZ_VALID,
                        epochs = EPOCHS,
                        callbacks= callbacks)
	return history

def predict_valid_img(valid_gen, model_json, model_h5):
	"""Predict age from validation data set.
	Input- valid gen, model.json, model.h5
	Output-None (Display predicted vs actual age) """

	model = cm.load_model(model_json, model_h5)

	test_X, test_Y  =  next(valid_gen)

	predict_valid = mean_age + std_age * (model.predict(test_X,
								batch_size = 32, verbose = True))

	actual_valid = mean_age + std_age*(test_Y)

	# Plot- precitons vs actual age using validation data
	fig, ax = plt.subplots(figsize = (7,7))
	ax.plot(actual_valid, predict_valid, 'r.', label = 'predictions')
	ax.plot(actual_valid, actual_valid, 'b-', label = 'actual')
	ax.legend(loc = 'upper right')
	ax.set_xlabel('Actual Age (Months)')
	ax.set_ylabel('Predicted Age (Months)')

	print('Mean absolute distance of the validation images: {} months'.
						format(round(np.mean(abs(actual_valid-
									list(flatten(predict_valid)))),1)))

	predict_valid_flat = list(flatten(predict_valid)) 
	residuals_flat = (actual_valid-predict_valid_flat)
	fig, axes = plt.subplots(figsize=(15, 3.5), nrows=1, ncols=3)
	axes[0].hist(predict_valid_flat);
	axes[0].set_xlabel("Predicted Age")
	axes[0].set_ylabel("Frequencey")
	axes[1].hist(actual_valid);
	axes[1].set_xlabel("Actual Age")
	axes[1].set_ylabel("Frequencey")
	axes[2].hist(residuals_flat);
	axes[2].set_xlabel("Residuals")
	axes[2].set_ylabel("Frequencey")
	plt.show()

if __name__ == "__main__":

	import exploration

	print("Tensorflow verison = {}".format (tensorflow.__version__))

	#Loading dataframe
	train_img_dir = os.path.join(os.path.dirname(__file__), 
								'mini_dataset/train/')
	valid_img_dir = os.path.join(os.path.dirname(__file__), 
								'mini_dataset/valid/')
	test_img_dir = os.path.join(os.path.dirname(__file__), 
								'mini_dataset/test/')

	# CSV file path + name          
	csv_train = os.path.join(os.path.dirname(__file__), 
								'mini_dataset/train.csv')
	csv_valid = os.path.join(os.path.dirname(__file__), 
								'mini_dataset/validation.csv')
	csv_test = os.path.join(os.path.dirname(__file__),
								 'mini_dataset/test.csv')

	df_mean_std_list = exploration.df_exploration(csv_train, csv_valid,
															 csv_test)

	train_df, valid_df, test_df = df_mean_std_list

	#reducing down the size of the image 
	IMG_SIZE = 256
	BATCH_SIZE = 32
	SEED = 42
	EPOCHS = 1
	TRAIN = False
	model_json = "model.json"
	model_h5 = "model.h5"

	train_gen, STEP_SIZE_TRAIN = get_input_train(train_img_dir, train_df,
									BATCH_SIZE, SEED, IMG_SIZE, 'id')

	valid_gen, STEP_SIZE_VALID = get_input_valid(valid_img_dir, valid_df,
								BATCH_SIZE, SEED, IMG_SIZE, 'Image ID')
	
	if TRAIN: 
		model = two_inputs_model(IMG_SIZE)

		history = train_model(model, train_gen, STEP_SIZE_TRAIN,
									valid_gen, STEP_SIZE_VALID, EPOCHS)
	
		pd.DataFrame.from_dict(history.history).to_csv('history.csv',
														index=False)
		plot_it(history)

	predict_valid_img(valid_gen, model_json, model_h5)