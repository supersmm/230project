import os

from PIL import Image
import numpy as np

def Create_filter(filename, out_dir, filter_name, is_image_filter = True):
	im = Image.open(filename).convert('LA') # Can be many different formats.
	pix = im.load()
	##repaint
	for i in range(0, im.size[0]):
		for j in range(0, im.size[1]):
			if (pix[i,j][0] < 40):
				pix[i,j] = (0, 255)  # Set the RGBA Value of the image (tuple)
			else:
				pix[i,j] = (255, 255)
	if is_image_filter:
		#Crop
		midHeight = im.size[1]>>1
		HRange = im.size[1]>>8
		for i in range(0, im.size[0]):
			for j in range(midHeight-HRange, midHeight+HRange):
				if pix[i,j][0] > 40:
					right = i - 1
					break	
		for i in range(0, im.size[0]):
			for j in range(midHeight-HRange, midHeight+HRange):
				if pix[im.size[0]-i-1, j][0] > 40:
					left = im.size[0] - i
					break		
		left, top, right, bottom = left, 0, right, im.size[1]
	else:
		left, top, right, bottom = 0, 0, im.size[0], im.size[1]
	image = im.crop((left, top, right, bottom))
	pix = image.load()
	for i in range(0, image.size[0]):
		for j in range(0, image.size[1]):
			p0 = pix[i,j][0] & pix[image.size[0]-i-1,j][0] & pix[i,image.size[1]-j-1][0]
			p1 = pix[i,j][1] | pix[image.size[0]-i-1,j][1] | pix[i,image.size[1]-j-1][1]
			pix[i,j] = (p0, p1)
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)
	image.save(os.path.join(out_dir, filter_name.split('\\')[-1]))  # Save the modified pixels

def apply_image_filter(filter_img, im):
	pix = im.load()
	midHeight = im.size[1]>>1
	HRange = im.size[1]>>8
	for i in range(0, im.size[0]):
		for j in range(midHeight-HRange, midHeight+HRange):
			if (pix[i,j][0] > 40):
				right = i - 1
				break	
	for i in range(0, im.size[0]):
		for j in range(midHeight-HRange, midHeight+HRange):
			if (pix[im.size[0]-i-1, j][0] > 40):
				left = im.size[0] - i
				break			
	#resize filter
	width = right - left
	Wpercent = float(width/float(filter_img.size[0]))
	height = int(float(filter_img.size[1])*float(Wpercent))
	left, top, right, bottom = left, (im.size[1]-height)>>1, right, im.size[1]-((im.size[1]-height)>>1)
	filter_image = filter_img.resize((width, bottom-top), Image.BILINEAR)  ##bottom-top to round it up to match image size
	filter_pix = filter_image.load()
	#crop image
	image = im.crop((left, top, right, bottom))
	pix = image.load()
	#apply filter
	for i in range(0, image.size[0]):
		for j in range(0, image.size[1]):
			p0 = pix[i,j][0] & filter_pix[i,j][0]
			p1 = pix[i,j][1] | filter_pix[i,j][1]
			pix[i,j] = (p0, p1)
	return image
	

def resize_and_save(image, output_dir, disease="diabetes", index=0, width = 128, height=128):
	# Use bilinear interpolation instead of the default "nearest neighbor" method

	image = image.resize((width, height), Image.BILINEAR)
	label= ""
	if disease == "diabetes":
		label = "01" 
	elif disease == "glaucoma":
		label = "10" 
	else:
		label = "00"
	rename = "GlaucomaVSDiabetes" + str(index) + "_(" + label + ")" + ".png"
	image = prenorm(image)
	image.save(os.path.join(output_dir, rename.split('\\')[-1])) # split('/') if using Mac OS

def prenorm(im):
	pix = im.load()
	max_W = 0
	for i in range(0, im.size[0]):
		for j in range(0, im.size[1]):
			if max_W < pix[i,j][0]: 
				max_W = pix[i,j][0]
	for i in range(0, im.size[0]):
		for j in range(0, im.size[1]):
			p = int(pix[i,j][0]/max_W*255)
			pix[i,j] = (p, pix[i,j][1])
	return im
	
if __name__ == '__main__':

	image_filter = Image.open('image_filter.png')
	im = Image.open("grayscale.png")
	im = apply_image_filter(image_filter, im)
	im.save("filtered.png")

