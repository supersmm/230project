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

Height = 128

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/GlaucomaVSDiabetes', help="Directory with the raw fundus dataset")
parser.add_argument('--output_dir', default='data/ResizedData', help="Where to write the new data")


def resize_and_save(filename, output_dir, disease="diabetes", index=0, height=Height):
	"""Resize the image contained in `filename` and save it to the `output_dir`"""
	image = Image.open(filename)
	# Use bilinear interpolation instead of the default "nearest neighbor" method
	hpercent = (height/float(image.size[1]))
	width = int(float(image.size[0])*float(hpercent))
	image = image.resize((width, height), Image.BILINEAR)
	label = ""
	if disease == "diabetes":
		label = "_0_1"
	else:
		label = "_1_0"
	rename = "GlaucomaVSDiabetes" + label + "(" + str(index) + ")" + ".jpg"
	image.save(os.path.join(output_dir, rename.split('\\')[-1])) # split('/') if using Mac OS


if __name__ == '__main__':
	args = parser.parse_args()

	assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

	for disease in ['diabetes', 'glaucoma']:
		# Define the data directories
		data_directory = os.path.join(args.data_dir, disease)

		filenames = os.listdir(data_directory)
		filenames = [os.path.join(data_directory, f) for f in filenames if f.endswith('.jpg') or f.endswith('.tif')]

		output_directory = os.path.join(args.output_dir, disease)

		if not os.path.exists(output_directory):
			os.mkdir(output_directory)
		else:
			print("Warning: output dir {} already exists".format(output_directory))

		print("Resizing {} data, saving preprocessed data to {}".format(disease,output_directory))
		index = 1
		for filename in tqdm(filenames):
			resize_and_save(filename, output_directory, disease, index, Height)
			index = index + 1

	print("Done building dataset")
