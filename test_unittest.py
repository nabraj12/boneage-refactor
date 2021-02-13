import unittest
import exploration
import train
import os


class TestCalc(unittest.TestCase):

	def test_exploration(self):

		self.assertEqual(len(train_df.index), 12611)
		self.assertEqual(len(valid_df.index), 1425)
		self.assertEqual(len(test_df.index), 200)
	
	def test_training(self):
		self.assertEqual(STEP_SIZE_TRAIN, 13)
		self.assertEqual(STEP_SIZE_VALID, 2)

if __name__ == '__main__':

	IMG_SIZE = 256
	BATCH_SIZE = 32
	SEED = 42
	EPOCHS = 1	
	pklfile = "mean_std_age.pkl"
	MODEL_JSON = "model.json"
	MODEL_H5 = "model.h5"

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

	df_list = exploration.df_exploration(csv_train, csv_valid, csv_test)
	train_df, valid_df, test_df = df_list
	
	train_gen, STEP_SIZE_TRAIN = train.get_input_train(train_img_dir,
									train_df, BATCH_SIZE,
									SEED, IMG_SIZE, 'id')
	valid_gen, STEP_SIZE_VALID = train.get_input_valid(valid_img_dir,
									valid_df, BATCH_SIZE,
									SEED, IMG_SIZE, 'Image ID')
	unittest.main()