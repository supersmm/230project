import os

from PIL import Image

def Create_filter(filename, out_dir, filter_name, is_image_filter = True):
	im = Image.open(filename) # Can be many different formats.
	pix = im.load()
	##repaint
	for i in range(0, im.size[0]):
		for j in range(0, im.size[1]):
			if (pix[i,j][0] < 40) and (pix[i,j][1] < 40) and (pix[i,j][2] < 40):
				pix[i,j] = (0, 0, 0)  # Set the RGBA Value of the image (tuple)
			else:
				pix[i,j] = (255, 255, 255)
	if is_image_filter:
		#Crop
		midHeight = im.size[1]>>1
		HRange = im.size[1]>>8
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
		left, top, right, bottom = left, 0, right, im.size[1]
	else:
		left, top, right, bottom = 0, 0, im.size[0], im.size[1]
	image = im.crop((left, top, right, bottom))
	pix = image.load()
	for i in range(0, image.size[0]):
		for j in range(0, image.size[1]):
			p0 = pix[i,j][0] & pix[image.size[0]-i-1,j][0] & pix[i,image.size[1]-j-1][0]
			p1 = pix[i,j][1] & pix[image.size[0]-i-1,j][1] & pix[i,image.size[1]-j-1][1]
			p2 = pix[i,j][2] & pix[image.size[0]-i-1,j][2] & pix[i,image.size[1]-j-1][2]
			pix[i,j] = (p0, p1, p2)
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)
	image.save(os.path.join(out_dir, filter_name.split('\\')[-1]))  # Save the modified pixels

def apply_image_filter(filter_img, im):
	pix = im.load()
	midHeight = im.size[1]>>1
	HRange = im.size[1]>>8
	for i in range(0, im.size[0]):
		for j in range(midHeight-HRange, midHeight+HRange):
			if (pix[i,j][0] > 40) or (pix[i,j][1] > 40) or (pix[i,j][2] > 40):
				right = i - 1
				break	
	for i in range(0, im.size[0]):
		for j in range(midHeight-HRange, midHeight+HRange):
			if (pix[im.size[0]-i-1, j][0] > 40) or (pix[im.size[0]-i-1,j][1] > 40) or (pix[im.size[0]-i-1,j][2]) > 40:
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
			p1 = pix[i,j][1] & filter_pix[i,j][1]
			p2 = pix[i,j][2] & filter_pix[i,j][2]
			pix[i,j] = (p0, p1, p2)
	return image

def apply_fundus_filter(filter_image, image):
	filter_pix = filter_image.load()
	pix = image.load()
	#apply filter
	for i in range(0, image.size[0]):
		for j in range(0, image.size[1]):
			p0 = pix[i,j][0] & filter_pix[i,j][0]
			p1 = pix[i,j][1] & filter_pix[i,j][1]
			p2 = pix[i,j][2] & filter_pix[i,j][2]
			pix[i,j] = (p0, p1, p2)
	return image
	

def resize_and_save(image, output_dir, disease="diabetes", index=0, width = 128, height=128):
	# Use bilinear interpolation instead of the default "nearest neighbor" method

	image = image.resize((width, height), Image.BILINEAR)
	label = ""
	if disease == "diabetes":
		label = "01"
	elif disease == "glaucoma":
		label = "10"
	else:
		label = "00"
	rename = "GlaucomaVSDiabetes" + str(index) + "_(" + label + ")" + ".jpg"
	image.save(os.path.join(output_dir, rename.split('\\')[-1])) # split('/') if using Mac OS

def prenorm(im):
	pix = im.load()
	max_R = 0
	max_G = 0
	max_B = 0
	for i in range(0, im.size[0]):
		for j in range(0, im.size[1]):
			if max_R < pix[i,j][0]: 
				max_W = pix[i,j][0]
			if max_G < pix[i,j][1]: 
				max_W = pix[i,j][1]
			if max_B < pix[i,j][2]: 
				max_W = pix[i,j][2]
	for i in range(0, im.size[0]):
		for j in range(0, im.size[1]):
			p0 = int(pix[i,j][0]/max_R*255)
			p1 = int(pix[i,j][1]/max_G*255)
			p2 = int(pix[i,j][2]/max_B*255)
			pix[i,j] = (p0, p1, p2)
	return im
	

#Create_filter("data/GlaucomaVSDiabetes/glaucoma/GlaucomaVSDiabetes_1_0 (1).tif",'data/GlaucomaVSDiabetes/filter')
'''im = Image.open("data/GlaucomaVSDiabetes/glaucoma/GlaucomaVSDiabetes_1_0 (1).tif")
filter_img = Image.open('data/filter/filter.jpg')
print(filter_img.size)
image = apply_image_filter(filter_img, im)
print(filter_img.size)
height = 128
hpercent = float(height/float(filter_img.size[1]))
width = int(float(filter_img.size[0])*float(hpercent))
image = image.resize((width, height), Image.BILINEAR)
image.save('img.jpg')'''


'''Create_filter("data/ResizedData/train_data/GlaucomaVSDiabetes1_(01).jpg",'data/filter', 'fundus_filter.jpg', is_image_filter = False)
filter_img = Image.open('data/filter/fundus_filter.jpg')
im = Image.open("data/ResizedData/train_data/GlaucomaVSDiabetes1_(01).jpg")
image = apply_fundus_filter(filter_img, im)
image.save('mymy.jpg')'''
