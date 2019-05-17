from PIL import Image
import numpy as np

def prenorm(im):
	pix = im.load()
	n_C = int(len(pix[0,0]))
	max_C = np.zeros(n_C,dtype=int)
	p = np.zeros(n_C,dtype=int)
	for i in range(0, im.size[0]):
		for j in range(0, im.size[1]):
			for c in range(0, n_C):
				if max_C[c] < pix[i,j][c]: 
					max_C[c] = pix[i,j][c]
	for i in range(0, im.size[0]):
		for j in range(0, im.size[1]):
			for c in range(0, n_C):
				p[c] = int(pix[i,j][c]/max_C[c]*255)
			pix[i,j] = tuple(p)
	return im

if __name__ == '__main__':

	im = Image.open("grayscale.png")
	prenorm(im)
	im.save("filtered.png")
