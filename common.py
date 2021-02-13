
from tensorflow.keras.models import model_from_json
import os
import sys
import pickle
import logging

def log_handler():
	"""Create a logger.
	Input- None
	Output-logger"""
	#prepare logger
	logger = logging.getLogger(__name__)
	logger.setLevel(logging.DEBUG)

	formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

	file_handler = logging.FileHandler('log.log')
	file_handler.setLevel(logging.ERROR)
	file_handler.setFormatter(formatter)

	logger.addHandler(file_handler)

	return logger


def load_model(model_json, model_h5):
	"""Load the model.
	Input- model json, model h5
	Output-model """

	logger = log_handler()

	if not os.path.isfile(model_json):
		logger.error("path {} does not exit".format(model_json))
		sys.exit("---model.json--- does not exits")

	if not os.path.isfile(model_h5):
		logger.error("path {} does not exit".format(model_h5))
		sys.exit("---model.h5--- does not exits")

	with open(model_json, "r") as json_file:
		loaded_model_json = json_file.read()

	model = model_from_json(loaded_model_json)

	# load weights into new model

	model.load_weights(model_h5)

	model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])

	return model

def mean_std_age(pickle_file_name):
	"""Get mean and std age use for Z-score calculation.
	Input- model json, model h5
	Output-model """

	logger = log_handler()

	if not os.path.isfile(pickle_file_name):
		logger.error("path {} does not exit".format(pickle_file_name))
		sys.exit("---mean_std_age.pkl--- does not exits")

	with open(pickle_file_name, 'rb') as f:
		mean_age, std_age  = pickle.load(f)

	return mean_age, std_age