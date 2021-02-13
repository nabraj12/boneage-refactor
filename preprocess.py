# Import required modules=====================
from skimage.color import rgb2gray
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
import multiprocessing as mp 
import timeit
import os
import logging
import warnings
warnings.filterwarnings("ignore")
#===========================================
# Input parameters===========================
global n_CLUSTERS
n_CLUSTERS = 6
global IMG_SIZE
IMG_SIZE = 256
#============================================
import common as cm
logger=cm.log_handler()

def image_segmentain(img_flat):

	"""Groups pixels using cluster using Kmeans.
	Inputs- 2D image
	Outputs-center of the clusters """
	kmeans = KMeans(n_clusters = n_CLUSTERS, random_state = 0).\
											fit(img_flat.reshape(-1,1))  
	"""Kmeans lables had issue with masking so center of each cluster
	is assigned for corresponding labels"""

	kmeans_centers = kmeans.cluster_centers_[kmeans.labels_]

	return  kmeans_centers.flatten()

def image_mask(kmeans_labels, img_gray_orig):

	"""Mask images based on clustering.
	Inputs- kmeans centes, 2D image (original)
	Outputs-bone-image, mask-imgage"""

	mask_img = np.zeros((img_gray_orig.shape[0], img_gray_orig.shape[1]))

	kmeans_labels_arr = kmeans_labels.reshape(img_gray_orig.shape[0],
											 img_gray_orig.shape[1])

	sort_labels = sorted(pd.Series(kmeans_labels).unique(),
													reverse = True)
	just_bone = ()

	if (np.sum(kmeans_labels_arr==sort_labels[0])) > 8000:
	    just_bone = np.where(kmeans_labels_arr==sort_labels[0])
	    mask_img[just_bone] = 1
		            
	if (np.sum(kmeans_labels_arr==sort_labels[1])) > 8000 and\
				 (np.sum(kmeans_labels_arr==sort_labels[1])) < 60000:
	    just_bone = np.where(kmeans_labels_arr==sort_labels[1])
	    mask_img[just_bone] = 1
	
	if (np.sum(kmeans_labels_arr==sort_labels[2]))>8000 and\
				 (np.sum(kmeans_labels_arr==sort_labels[2])) < 70000:
	    just_bone = np.where(kmeans_labels_arr==sort_labels[2])
	    mask_img[just_bone] = 1
	
	if (np.sum(kmeans_labels_arr==sort_labels[3]))>8000 and\
				(np.sum(kmeans_labels_arr==sort_labels[3])) < 70000:
	    just_bone = np.where(kmeans_labels_arr==sort_labels[3])
	    mask_img[just_bone] = 1
	
	if not just_bone:
		just_bone = np.where(kmeans_labels_arr==sort_labels[1]) 
		mask_img[just_bone] = 1

	return just_bone, mask_img

def img_resize(img, img_height):
	"""Resize image shape.
	Inputs- 2D image, intended height
	Outputs-resized image """

	img_width  = int(img_height*img.shape[1]/img.shape[0])

	img_pil = Image.fromarray(img) # convert array back to image

	img_pil_resize = img_pil.resize((img_width, img_height), Image.LANCZOS) # resize

	return np.array(img_pil_resize)

def img_pad_resize(img_just_bone, IMG_SIZE):
	"""First make image square and then resize it.
	Inputs- 2D image, New image size
	Outputs-Squred resized image """
  
	size_diff = img_just_bone.shape[0]-img_just_bone.shape[1]

	if size_diff > 0: # hieght is longer than width
		top = 0
		bottom = 0
		left = int(abs(size_diff)/2.)
		right = (abs(size_diff)-left)
	
	elif size_diff < 0: # hieght is shorter than width
		left = 0
		right = 0
		top = int(abs(size_diff)/2.)
		bottom = (abs(size_diff)-top)
	
	else:
		top = 0
		bottom = 0
		left = 0
		right = 0

	img_bone_square = np.pad (img_just_bone,((top,bottom),(left,right)), 'constant')

	img_bone = img_resize(img_bone_square, IMG_SIZE)

	return img_bone

def img_preprocess_core(img_gray_orig):
	"""Calls sever functions to perform one iteration of imaging
														 preprocessing.
	Inputs- 2D image (original)
	Outputs-2D preprocessed image """
    
	img_flat = img_gray_orig.reshape(img_gray_orig.shape[0] *
										 img_gray_orig.shape[1])
	     
	kmeans_labels =  image_segmentain(img_flat)

	kmeans_labels_arr = kmeans_labels.reshape(img_gray_orig.shape[0],
									 img_gray_orig.shape[1])

	just_bone, mask_img = image_mask (kmeans_labels, img_gray_orig)
	   
	img_clean_background = mask_img * img_gray_orig

	img_just_bone = img_clean_background[min(just_bone[0]):
					max(just_bone[0]),min(just_bone[1]):
					max(just_bone[1])]
	
	return img_just_bone

def img_preprocessing(save_path,img_path, filename):
	"""Call img_preprocess core twice for two iteration 
	Inputs- dir(to save preprocessed img), dir (original img), filename
	Outputs-None (preprocessed image is saved in  the img save dir) """

	save_path_filename = save_path + filename

	#Check if file exits
	if not os.path.exists(img_path + filename):
		logger.error(" image path {} does not exit".
								format(img_path + filename))

	image = plt.imread(img_path + filename)

	img_gray_orig_0 = rgb2gray(image)

	img_gray_orig = img_resize(img_gray_orig_0, 2*IMG_SIZE)

	img_just_bone = img_preprocess_core(img_gray_orig)

	try:
	    img_bone = img_pad_resize(img_just_bone, 2*IMG_SIZE) 
	     #Second iteration of image segmentation
	    img_just_bone = img_preprocess_core(img_bone)
	    img_bone = img_pad_resize(img_just_bone, IMG_SIZE)

	    plt.imsave(save_path_filename, img_bone)
	    
	except ValueError:
		logger.error("Unable to run 2nd interaton for {}".format(filename))
   
def multi_run_wrapper(args):
	"""Wrapper function to use multiprocessing module.
	Inputs- args
	Outputs- function call """
	return img_preprocessing(*args)

def main(save_dir, img_dir, df, fname_col):
	"""Call multi processing.
	Inputs- dir (to save img), dir (to original img), df, col name
	Outputs-None """
	pool = mp.Pool(mp.cpu_count())
	result = pool.map(multi_run_wrapper,[(save_dir, img_dir, 
						fname) for fname in df[fname_col].values[0:4]])
  
if __name__ == "__main__":

	import exploration

	# Original data directory- for image preprocessing
	train_img_pre = os.path.join(os.path.dirname(__file__), 
								'mini_dataset/train/')
	valid_img_pre = os.path.join(os.path.dirname(__file__), 
								'mini_dataset/valid/')
	test_img_pre = os.path.join(os.path.dirname(__file__), 
								'mini_dataset/test/')
	# Save pre-processed images
	train_save_dir = os.path.join(os.path.dirname(__file__), 
								'mini_dataset/train_pre/')
	valid_save_dir = os.path.join(os.path.dirname(__file__), 
								'mini_dataset/valid_pre/')
	test_save_dir = os.path.join(os.path.dirname(__file__), 
								'mini_dataset/test_pre/')
	# CSV file path + name          
	csv_train = os.path.join(os.path.dirname(__file__), 
								'mini_dataset/train.csv')
	csv_valid = os.path.join(os.path.dirname(__file__), 
								'mini_dataset/validation.csv')
	csv_test = os.path.join(os.path.dirname(__file__),
								 'mini_dataset/test.csv')

	df_list = exploration.df_exploration(csv_train, csv_valid,
												 	csv_test)

	train_df, valid_df, test_df = df_list
	
	start = timeit.default_timer()
	main(train_save_dir, train_img_pre,train_df, "id")
	main(valid_save_dir, valid_img_pre,valid_df, "Image ID")
	main(test_save_dir, test_img_pre,test_df, "Case ID")

	stop = timeit.default_timer()
	print('Time in (sec): ', stop - start)  