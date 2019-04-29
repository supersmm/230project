import os

from PIL import Image
from tqdm import tqdm

def get_image_ratio(filename, Ver_range = 8):##Ver_range 0-->10 the bigger Ver_range is the fast function will run, but less accurate
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

def get_biggest_Image_Ratio(data_dir='data/GlaucomaVSDiabetes'):
	ratio = 0
	imagename = ""
	assert os.path.isdir(data_dir), "Couldn't find the dataset at {}".format(data_dir)

	for disease in ['diabetes', 'glaucoma']:
		# Define the data directories
		data_directory = os.path.join(data_dir, disease)

		filenames = os.listdir(data_directory)
		filenames = [os.path.join(data_directory, f) for f in filenames if f.endswith('.jpg') or f.endswith('.tif')]

		print("Finding Image with biggest ratio from {}".format(disease))
		for filename in tqdm(filenames):
			l_ratio = get_image_ratio(filename)
			if l_ratio > ratio:
				ratio = l_ratio
				imagename = filename

	print("Found {} with biggest ratio".format(imagename))
	return imagename

def get_fundus_ratio(filename):##Ver_range 0-->10 the bigger Ver_range is the fast function will run, but less accurate
	im = Image.open(filename) # Can be many different formats.
	pix = im.load()
	left = 0
	right = 0
	for i in range(0, im.size[0]):
		if pix[i,0][0] > 40:
			if pix[i,0][1] > 40:
				if pix[i,0][2] > 40:
					right = i - 1
					break	
	for i in range(0, im.size[0]):
		if pix[im.size[0]-i-1, 0][0] > 40:
			if pix[im.size[0]-i-1,0][1] > 40:
				if pix[im.size[0]-i-1,0][2] > 40:
					left = im.size[0] - i
					break	
	Width = right - left
	if (right == 1) or (left == 1):
		os.remove(filename)  ##remove image that's cropped horizenally
		Width = 100*im.size[0]
	return Width/im.size[0]

def get_smallest_Image_Ratio(data_dir = "data/ResizedData"):
	ratio = 100
	imagename = ""
	assert os.path.isdir(data_dir), "Couldn't find the dataset at {}".format(data_dir)

	for disease in ['train_data', 'val_data', 'test_data']:
		# Define the data directories
		data_directory = os.path.join(data_dir, disease)
		if os.path.exists(data_directory):
			filenames = os.listdir(data_directory)
			filenames = [os.path.join(data_directory, f) for f in filenames if f.endswith('.jpg')]

			print("Finding Image with smallest fundus ratio from {}".format(disease))
			for filename in tqdm(filenames):
				l_ratio = get_fundus_ratio(filename)
				if l_ratio < ratio:
					ratio = l_ratio
					imagename = filename

	print("Found {} with biggest ratio".format(imagename))
	return imagename

#ratio, filename = get_biggest_Ratio()
#print("Biggest ratio is {} in image {}".format(ratio, filename))
#ratio=get_ratio(filename, Ver_range = 4) ##get a more accurate ratio of the selected image
#Create_filter("data/GlaucomaVSDiabetes/glaucoma/GlaucomaVSDiabetes_1_0 (1).tif",'data/GlaucomaVSDiabetes/filter')
#print(str(255|0))
#get_smallest_Image_Ratio()
