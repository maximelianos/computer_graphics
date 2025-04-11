import numpy as np
import skimage
from skimage import io
from skimage import filters
from skimage.restoration import denoise_bilateral

'''
fin = open('hi.ppm', 'r')
_p3 = fin.readline()
m, n = map(int, fin.readline().split())
_maxval = fin.readline()

img = np.zeros((n, m, 3))
for i in range(n):
	if i % 10 == 0:
		print(i)
	for j in range(m):
		r, g, b = map(int, fin.readline().split())
		img[i, j] = np.array([r, g, b])
io.imsave('out.png', img.astype(np.uint8))
'''

img = io.imread('hi.bmp')

img = skimage.filters.gaussian(img, sigma=1, multichannel=True)
#img = denoise_bilateral(img, sigma_color=10.25, sigma_spatial=15,
#                multichannel=True)
io.imsave('out.bmp', (img * 255).astype(np.uint8))
