"""Split our Fundus dataset into train/val/test and resize images to (192, 128).

Our Fundus dataset comes into the following format:
    diabetes/
        GlaucomaVSDiabetes_1_0 (1).jpg
        GlaucomaVSDiabetes_1_0 (2).jpg
        ...
    glaucoma/
        GlaucomaVSDiabetes_0_1 (1).jpg
        GlaucomaVSDiabetes_0_1 (2).jpg
        ...

Original images have various sizes including (2743, 1936), (2376, 1584), etc.
Resizing to (192, 128) reduces the dataset size from 3.3 MB to 6-60 KB, and loading smaller images
makes training faster.
"""

import argparse
import random
import os

from PIL import Image
from tqdm import tqdm
from Get_Image_Ratio import get_biggest_Image_Ratio
from Filter import *

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../data/GlaucomaVSDiabetes', help="Directory with the raw fundus dataset")
parser.add_argument('--output_dir', default='../data/ResizedData', help="Where to write the new data")
parser.add_argument('--filter_dir', default='../data/filter', help="Where to save and get the image filter")

args = parser.parse_args()

def resize_images(filter_img, width= 128, height = 128):
	
	assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)
	for disease in ['diabetes', 'glaucoma', 'healthy']:
		# Define the data directories
		data_directory = os.path.join(args.data_dir, disease)
		if not os.path.exists(data_directory):
			continue
		# Get the filenames in each directory (train and test)
		filenames = os.listdir(data_directory)
		filenames = [os.path.join(data_directory, f) for f in filenames if f.endswith('.jpg') or f.endswith('.tif')]

		# Split the images into 80% train, 10% val, 10% test.
		# Make sure to always shuffle with a fixed seed so that the split is reproducible
		random.seed(230)
		filenames.sort()
		random.shuffle(filenames)

		num_images = len(filenames)
		split1 = int(0.8 * num_images)
		split2 = int(0.9 * num_images)
		train_filenames = filenames[:split1]
		val_filenames = filenames[split1:split2]
		test_filenames = filenames[split2:]

		filenames = {'train': train_filenames,
			     'val': val_filenames,
			     'test': test_filenames}
		if not os.path.exists(args.output_dir):
			os.mkdir(args.output_dir)


		# Preprocess train, val and test
		for split in ['train', 'val', 'test']:
			output_dir_split = os.path.join(args.output_dir, '{}_data'.format(split))
			if not os.path.exists(output_dir_split):
				os.mkdir(output_dir_split)
			else:
				print("Warning: dir {} already exists".format(output_dir_split))

			print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
			index = 1
			for filename in tqdm(filenames[split]):
				image = Image.open(filename).convert('LA')
				try:
					image = apply_image_filter(filter_img, image)
					resize_and_save(image, output_dir_split, disease, index, width = width, height = height)
					index = index + 1
				except NameError:
					os.remove(filename) ##exception raise on image being too dark

	

if __name__ == '__main__':
	
	image_filter = 'image_filter.png'
	
	filename = get_biggest_Image_Ratio(args.data_dir)
	
	Create_filter(filename, args.filter_dir, image_filter)
	filter_img = Image.open(os.path.join(args.filter_dir, image_filter.split('\\')[-1]))
	height = 224
	#hpercent = float(height/float(filter_img.size[1]))
	width = 224
	resize_images(filter_img, width = width, height = height)
	
	print("Done building dataset")
