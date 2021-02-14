# boneage-refactor

1) Required directories and files

	├───boneage-refactor                                 
		├───exploration.py

		├───preprocess.py

		├───checking.py
		
		├───train.py
		├───predict.py
		├───main.py
		├───common.py
		├───test_unittest.py
		├───mini_dataset
		│   ├───test
		│	│	├───orignal test png images(required)
		│   ├───test_pre
		│	│	├───BLANK Directory(preprocessed images will be stored)
		│   ├───train
		│	│	├───orignal train png images(required)
		│   ├───train_pre
		│	│	├───BLANK Directory(preprocessed images will be stored)
		│   ├───valid
		│	│	├───orignal validation png images(required)
		│   └───valid_pre
		│	│	├───BLANK Directory(preprocessed images will be stored)
	   	│	├───train.csv(required)
		│   ├───valid.csv(required)
		│   └───test.csv(required)

2) From the terminal- run the following code

		python main.py -i INPUT1 -c INPUT2 -t INPUT3 -v INPUT4 -p INPUT5

		INPUT1: True or False- True- if image preprocessing is required
		INPUT2: True or False- True- if images need to be checked for missing or corruption
		INPUT3: True or False- True- if model training is required
		INPUT4: True or False- True- if prediction is needed on validation images
		INPUT5: True or False- True- if prediction on test images is required
3) For help: python main.py -h

