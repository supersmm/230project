import os

from PIL import Image

def Create_filter(filename, out_dir):
	im = Image.open(filename) # Can be many different formats.
	pix = im.load()
	##repaint
	for i in range(0, im.size[0]):
		for j in range(0, im.size[1]):
			if (pix[i,j][0] < 40) and (pix[i,j][1] < 40) and (pix[i,j][2] < 40):
				pix[i,j] = (0, 0, 0)  # Set the RGBA Value of the image (tuple)
			else:
				pix[i,j] = (255, 255, 255)
	#Crop
	for i in range(0, im.size[0]):
		for j in range(0, im.size[1]):
			if pix[i,j][0] > 40:
				if pix[i,j][1] > 40:
					if pix[i,j][2] > 40:
						right = i - 1
						break	
	for i in range(0, im.size[0]):
		for j in range(0, im.size[1]):
			if pix[im.size[0]-i-1, j][0] > 40:
				if pix[im.size[0]-i-1,j][1] > 40:
					if pix[im.size[0]-i-1,j][2] > 40:
						left = im.size[0] - i
						break		
	left, top, right, bottom = left, 0, right, im.size[1]
	rename = "filter"
	image = im.crop((left, top, right, bottom))
	pix = image.load()
	for i in range(0, image.size[0]):
		for j in range(0, im.size[1]):
			p0 = pix[i,j][0] & pix[image.size[0]-i-1,j][0]
			p1 = pix[i,j][1] & pix[image.size[0]-i-1,j][1]
			p2 = pix[i,j][2] & pix[image.size[0]-i-1,j][2]
			pix[i,j] = (p0, p1, p2)
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)
	image.save(os.path.join(out_dir, rename.split('\\')[-1]))  # Save the modified pixels