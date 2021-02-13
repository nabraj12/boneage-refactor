
import matplotlib.image as mpimg
import os.path
import matplotlib.pyplot as plt
import logging
import sys
#==============================
#prepare logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('log.log')
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

def check_image(df, fname_col, img_dir):
	"""Check for missing/corrupted images.
	Inputs- df, col name contains file name, image dir
	Outputs- succees status """

	for filename in df[fname_col].values[0:4]:

		if not os.path.isfile(img_dir+filename):
			logger.error("path {} does not exit".format(img_dir+filename))
			success = False
		else:
			try:
				img = mpimg.imread(img_dir + filename)
				success = True
			except OSError:
				success = False
				logger.error("image is {} corrupted/missing".
													format(filename))
				
	return success

def display_image(df, fname_col, img_dir, n):
	"""Displays train, valid, and test images.
	Inputs- df, col-name contains file anme, img dir, # of imgs to display
	Outputs-None(display images) """
	# Display some train images
	nrows = 1+n//20 
	fig, axs = plt.subplots(nrows,20, figsize=(20,1.2*nrows),
						 facecolor='w', edgecolor='k')
	axs = axs.ravel()

	for idx, filename in enumerate (df[fname_col][0:n].values):

		if not os.path.isfile(img_dir+filename):
			logger.error("path {} does not exit".format(img_dir+filename))
						
		img = mpimg.imread(img_dir + filename)

		axs[idx].imshow(img)
		axs[idx].set_axis_off()
	    
	plt.subplots_adjust(wspace=0, hspace=0)
	plt.show()


if __name__ == "__main__":

	import exploration
	# Image directory to check whether any image file is..
									# missing or corrupted
	train_img_pre = os.path.join(os.path.dirname(__file__), 
								'mini_dataset/train/')
	valid_img_pre = os.path.join(os.path.dirname(__file__), 
								'mini_dataset/valid/')
	test_img_pre = os.path.join(os.path.dirname(__file__), 
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

	train_df,valid_df, test_df = df_mean_std_list
	check = check_image(train_df, 'id', train_img_pre)

	if check: print("No missing or corrupted image found.")
	else: print ("Image directory contains missing or corrupted image")

	# Display images
	display_image(train_df, 'id', train_img_pre, 4)
	display_image(valid_df, 'Image ID', valid_img_pre, 4)
	display_image(test_df, 'Case ID', test_img_pre, 4)

