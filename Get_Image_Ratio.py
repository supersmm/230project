import argparse
import os

from PIL import Image
from tqdm import tqdm

def get_ratio(filename, Ver_range = 8):##Ver_range 0-->10 the bigger Ver_range is the fast function will run, but less accurate
	im = Image.open(filename) # Can be many different formats.
	pix = im.load()
	left = 0
	right = 0
	midHeight = im.size[1]>>1
	HRange = im.size[1]>>Ver_range
	for i in range(0, im.size[0]):
		for j in range(midHeight-HRange, midHeight+HRange):
			if pix[i,j][0] > 40:
				if pix[i,j][1] > 40:
					if pix[i,j][2] > 40:
						right = i - 1
						break	
	for i in range(0, im.size[0]):
		for j in range(midHeight-HRange, midHeight+HRange):
			if pix[im.size[0]-i-1, j][0] > 40:
				if pix[im.size[0]-i-1,j][1] > 40:
					if pix[im.size[0]-i-1,j][2] > 40:
						left = im.size[0] - i
						break	
	Width = right - left
	if (right == 1) or (left == 1):
		os.remove(filename)  ##remove image that's cropped horizenally
		Width = 0
	Height = im.size[1]
	return Width/Height

def get_biggest_Ratio():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', default='data/GlaucomaVSDiabetes', help="Directory with the raw fundus dataset")
	
	ratio = 0
	imagename = ""
	args = parser.parse_args()
	assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

	for disease in ['diabetes', 'glaucoma']:
		# Define the data directories
		data_directory = os.path.join(args.data_dir, disease)

		filenames = os.listdir(data_directory)
		filenames = [os.path.join(data_directory, f) for f in filenames if f.endswith('.jpg') or f.endswith('.tif')]

		print("Finding Image with biggest ratio")
		for filename in tqdm(filenames):
			l_ratio = get_ratio(filename)
			if l_ratio > ratio:
				ratio = l_ratio
				imagename = filename

	print("Found Image with biggest ratio")
	return ratio, imagename

#ratio, filename = get_biggest_Ratio()
#print("Biggest ratio is {} in image {}".format(ratio, filename))
#ratio=get_ratio(filename, Ver_range = 4) ##get a more accurate ratio of the selected image
Create_filter("data/GlaucomaVSDiabetes/glaucoma/GlaucomaVSDiabetes_1_0 (1).tif",'data/GlaucomaVSDiabetes/filter')
#print(str(255|0))
