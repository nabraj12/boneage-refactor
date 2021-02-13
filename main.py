import pickle
import os
import timeit
import pandas as pd
import getopt
import sys
#===============================
import exploration
import preprocess
import checkimg
import train
import predict
import common as cm
import common as cm
logger=cm.log_handler()

def main():
	"""Main function to call all other functions.
	Input- None
	Output-None"""

	# exploration.py
	df_list = exploration.df_exploration(csv_train, csv_valid, csv_test)
	train_df, valid_df, test_df = df_list
	exploration.my_plots(train_df)
	print("======Data Exploration Completed==================")

	#checkimg.py
	if CHECK_IMG:
		check = checkimg.check_image(train_df, 'id', train_img_pre)
		if check: 
			print("No missing or corrupted image found.")
		else: 
			print ("Image directory contains missing or corrupted image")
		print("======Image checkimg completed==================")

	#preprocess.py
	if PREPROCESS_REQ:
		start = timeit.default_timer()
		preprocess.main(train_save_dir, train_img_pre,train_df, "id")
		preprocess.main(valid_save_dir, valid_img_pre,valid_df, "Image ID")
		preprocess.main(test_save_dir, test_img_pre,test_df, "Case ID")
		stop = timeit.default_timer()
		print('Time in (sec): ', stop - start)
		print("======Image Preprocessing Completed==================")
	
	# Display images
	checkimg.display_image(train_df, 'id', train_img_dir, 50)
	checkimg.display_image(valid_df, 'Image ID', valid_img_dir, 50)
	checkimg.display_image(test_df, 'Case ID', test_img_dir, 50)

	#train.py
	if TRAIN or VALID:
		train_gen, STEP_SIZE_TRAIN = train.get_input_train(train_img_dir,
										train_df, BATCH_SIZE,
										SEED, IMG_SIZE, 'id')
		valid_gen, STEP_SIZE_VALID = train.get_input_valid(valid_img_dir,
										valid_df, BATCH_SIZE,
										SEED, IMG_SIZE, 'Image ID')		
		print (STEP_SIZE_TRAIN, STEP_SIZE_VALID)
		if TRAIN: 
			model = train.two_inputs_model(IMG_SIZE)
			history = train.train_model(model, train_gen, STEP_SIZE_TRAIN,
									valid_gen, STEP_SIZE_VALID, EPOCHS)	
			pd.DataFrame.from_dict(history.history).to_csv('history.csv',
															index=False)
			train.plot_it(history)
		train.predict_valid_img(valid_gen, MODEL_JSON, MODEL_H5)
		print("======Model Training Completed==================")

	#predict.py
	if PREDICTION:
		model = cm.load_model(MODEL_JSON, MODEL_H5)
		test_XX, test_YY = predict.input_generator(test_img_dir, test_df,
									'Case ID', BATCH_SIZE, SEED,IMG_SIZE)	
		predict_age_month = predict.predict_bone_age(model, BATCH_SIZE,
									test_XX,test_YY, pklfile)
		predict.plot_prediction(test_YY, predict_age_month)

		print("======Age Prediction Completed==================")

if __name__== "__main__":

	PREPROCESS_REQ, CHECK_IMG, TRAIN, VALID, PREDICTION = [None]*5

	argv = sys.argv[1:]
	
	try:
		opts, args = getopt.getopt(argv,"hi:c:t:v:p:")

	except getopt.GetoptError:
		logger.error('Aruments were not given correctly')
		sys.exit("---incorrect arguments type main.py -h")

	for opt, arg in opts:
		if opt == '-h':
			print("===================================")
			print('main.py -i prepro -c imgcheck -t train -v valid -p predict')
			print('main.py -i False -c False -t Flase -v False -p True')
			print("===================================")
			sys.exit()
	
		elif opt in ("-i"):
			PREPROCESS_REQ = True if arg=='True' else False		
		elif opt in ("-c"):
	 		CHECK_IMG = True if arg=='True' else False		
		elif opt in ("-t"):
	 		TRAIN = True if arg=='True' else False
		elif opt in ("-v"):
	 		VALID = True if arg=='True' else False
		elif opt in ("-p"):
	 		PREDICTION = True if arg=='True' else False
	
	if None in [PREPROCESS_REQ, CHECK_IMG, TRAIN, VALID, PREDICTION]:
		logger.error('Aruments were not given correctly')
		sys.exit("---incorrect arguments type main.py -h")

	IMG_SIZE = 256
	BATCH_SIZE = 32
	SEED = 42
	EPOCHS = 1	
	pklfile = "mean_std_age.pkl"
	MODEL_JSON = "model.json"
	MODEL_H5 = "model.h5"

	with open(pklfile, 'rb') as f:
		mean_age, std_age  = pickle.load(f)

	# Original data directory- for image preprocessing
	train_img_pre = os.path.join(os.path.dirname(__file__), 
								'mini_dataset/train/')
	valid_img_pre = os.path.join(os.path.dirname(__file__), 
								'mini_dataset/valid/')
	test_img_pre = os.path.join(os.path.dirname(__file__), 
								'mini_dataset/test/')

	# Directory to save pre-processed images
	train_save_dir = os.path.join(os.path.dirname(__file__), 
								'mini_dataset/train_pre/')
	valid_save_dir = os.path.join(os.path.dirname(__file__), 
								'mini_dataset/valid_pre/')
	test_save_dir = os.path.join(os.path.dirname(__file__), 
								'mini_dataset/test_pre/')

	#Image directories for model training
	train_img_dir = train_save_dir
	valid_img_dir = valid_save_dir
	test_img_dir  = test_save_dir
	# CSV file path + name          
	csv_train = os.path.join(os.path.dirname(__file__), 
								'mini_dataset/train.csv')
	csv_valid = os.path.join(os.path.dirname(__file__), 
								'mini_dataset/validation.csv')
	csv_test = os.path.join(os.path.dirname(__file__),
								 'mini_dataset/test.csv')

	main()
	print("======Task Completed==================")



