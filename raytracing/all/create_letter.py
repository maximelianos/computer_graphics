import numpy as np
import skimage
from skimage import io
from skimage import filters
from skimage.restoration import denoise_bilateral
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage import feature
from skimage import transform

from random import random
from random import shuffle
from random import seed

seed(0)

from matplotlib import pyplot as plt

def dist(p1, p2):
	return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def draw(img, p1, p2):
	if p1[1] > p2[1]:
		p1, p2 = p2, p1
	if p1[1] == p2[1]:
		for y in range(min(p1[0], p2[0]), max(p1[0], p2[0]) + 1):
			img[y, p1[1]] = np.array([0, 0, 1])
	else:
		for x in range(min(p1[1], p2[1]) + 1, max(p1[1], p2[1])):	
			y1 = (x - 1 - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
			y2 = (x + 1 - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
			y1 = int(y1)
			y2 = int(y2)
			for y in range(min(y1, y2), max(y1, y2)):
				img[y, x] = np.array([0, 0, 1])
			if y1 == y2:
				img[y1, x] = np.array([0, 0, 1])
		x1, x2 = min(p1[1], p2[1]), max(p1[1], p2[1])
		y1 = (x1 - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
		y2 = (x1 + 1 - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
		y1 = int(y1)
		y2 = int(y2)
		for y in range(min(y1, y2), max(y1, y2)):
			img[y, x1] = np.array([0, 0, 1])
		y1 = (x2 - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
		y2 = (x2 - 1 - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
		y1 = int(y1)
		y2 = int(y2)
		for y in range(min(y1, y2), max(y1, y2)):
			img[y, x2] = np.array([0, 0, 1])

def eq(pa, pb):
	
	return dist(pa, pb) <= 0.05

def sin_prod(a, b, c):
	# vectors b->a and b->c
	return (a[1] - b[1]) * (c[0] - b[0]) - (a[0] - b[0]) * (c[1] - b[1])

def cos_prod(a, b, c):
	# vectors b->a and b->c
	return (a[0] - b[0]) * (c[0] - b[0]) + (a[1] - b[1]) * (c[1] - b[1])

def small_tri(a, b, c):
	return abs(sin_prod(a, b, c) / 2) < 0.1

def intersect(a, b, c, d):
	res1_b = sin_prod(a, b, c)
	res2_b = sin_prod(a, b, d)
	res1_c = sin_prod(d, c, b)
	res2_c = sin_prod(d, c, a)
	res1 = sin_prod(a, b, c) * sin_prod(a, b, d)
	res2 = sin_prod(d, c, b) * sin_prod(d, c, a)
	if res1_b == 0 and res2_b == 0 and res1_c == 0 and res2_c == 0:
		seg1_d = cos_prod(a, d, b)
		seg1_c = cos_prod(a, c, b)
		seg2_b = cos_prod(d, b, c)
		seg2_a = cos_prod(d, a, c)
		if seg1_d >= 0 and seg1_c >= 0 and seg2_b >= 0 and seg2_a >= 0:
			return False
		else:
			return True
	res1 = res1 <= 0
	res2 = res2 <= 0
	return not(eq(a, c) or eq(a, d) or eq(b, c) or eq(b, d)) and res1 and res2

def save(img):
	if img.max() > 1.5:
		print("Save uint")
		io.imsave('out.bmp', (img.astype(np.uint8)))
	else:
		print("Save real")
		io.imsave('out.bmp', (np.clip(img, 0, 1)*255).astype(np.uint8))

def angle_map(img):
	h, w = img.shape[:2]
	edges_v = filters.sobel_h(img)
	edges_h = filters.sobel_v(img)
	length = (edges_v ** 2 + edges_h ** 2) ** 0.5
	av_length = np.average(length)
	tan_map = np.arctan2(edges_v, edges_h)
	tan_map[tan_map < 0] += np.pi * 2
	
	#for y in range(h):
	#	for x in range(w):
	
	window = 5
	h_bins = 40
	
	edge_map = np.zeros((h, w))
	edge_map[10:h - 10, 10:w - 10] = (length > av_length)[10:h - 10, 10:w - 10]
	result_measure = np.zeros((h, w))
	
	print('start')
	for y, x in zip(*edge_map.nonzero()):
		#print(y, x)
		y_1, y_2 = y - window, y + window + 1
		x_1, x_2 = x - window, x + window + 1
		#for yd in range(y - window, y + window + 1):
		#	for xd in range(x - window, x + window + 1):
		
		#print(edge_map[y_1:y_2, x_1:x_2])
		angles = tan_map[y_1:y_2, x_1:x_2][edge_map[y_1:y_2, x_1:x_2] == 1]
		lens = length[y_1:y_2, x_1:x_2][edge_map[y_1:y_2, x_1:x_2] == 1]
		lens = lens / np.sum(lens)
		
		h = np.histogram(angles, bins=h_bins, range=(0, np.pi * 2), weights=lens)
		h = h[0]
		left_percentile, right_percentile = 0.1, 0.1
		l, r = 0, h_bins - 1
		l_sum = h[l]
		r_sum = h[r]
		while l_sum < left_percentile:
			l += 1
			l_sum += h[l]
		while r_sum < right_percentile:
			r -= 1
			r_sum += h[r]
		
		l1, r1 = l, r
		
		angles += np.pi
		angles[angles > np.pi * 2] -= np.pi * 2
		
		h = np.histogram(angles, bins=h_bins, range=(0, np.pi * 2), weights=lens)
		h = h[0]
		l, r = 0, h_bins - 1
		l_sum = h[l]
		r_sum = h[r]
		while l_sum < left_percentile:
			l += 1
			l_sum += h[l]
		while r_sum < right_percentile:
			r -= 1
			r_sum += h[r]
		
		l2, r2 = l, r
		
		result_measure[y, x] = min(r1 - l1, r2 - l2)
		#print(y_1, y_2, x_1, x_2)
	print('end')
	#save(edge_map)
	
	print(result_measure.max())
	result_measure = filters.gaussian(result_measure, sigma=20, preserve_range=True)
	print(result_measure.max())
	save(result_measure)
	
	return result_measure
		

def process(filename, out_img, out_list):
	img = io.imread(filename)[:, :, 0]
	h, w = img.shape
	img = transform.rotate(img, 0.01)
	img_bw = filters.gaussian(img, sigma=1.5)
	precision = angle_map(img)
	
	
	inside_mask = img < 0.2
	
	inside_list = []
	for y in range(20, h - 20):
		for x in range(20, w - 20):
			if inside_mask[y, x]: # Is inside
				inside_list.append((y, x))
	shuffle(inside_list)
	
	free_mask = np.zeros(inside_mask.shape)
	free_mask[...] = inside_mask
	
	coords = corner_peaks(corner_harris(img, k=0.01, sigma=2, eps=2.0), min_distance=1)
	img = feature.canny(img)
	
	img_src = np.zeros(img.shape)
	img_src[...] = img
	h, w = img.shape
	
	
	edge_dense = np.zeros((h, w))
	#for y in range(1, h):
	#	for x in range(1, w):
	#		edge_dense[y, x] = img[y, x] + edge_dense[y - 1, x] + edge_dense[y, x - 1] - edge_dense[y - 1, x - 1]
			#print(edge_dense[y, x])
	
	#img = img > 0.5
	img = np.stack([img, img, img], axis=2).astype(np.float)
	
	n = 0
	edges = dict() # dict( [idx -> [y, x]], ... )
	edges_idc = dict() # dict( [idx -> [idx1, idx2, ...]], ... )
	
	eps = 10 # Pixels
	max_r = eps * 1.3
	int_r = int(max_r) + 1
	k_scale = 1
	
	edge_mask = np.zeros((h, w))
	points = []
	edge_chain = []
	graph = dict() # dict( [point -> [point, point, ...], ... )
	edge_coors = dict() # dict( [point -> idx], ... )
	pure_edge = set()

	edge_list = []
	for y in range(20, h - 20):
		for x in range(20, w - 20):
			if img[y, x, 1] != 0: # Is an edge
				edge_list.append((y, x))
	shuffle(edge_list)
	
	sobel_eps = 4
	
	mn = 10000
	mx = 0
	for p in edge_list:
		y, x = p
		if img[y, x, 1] != 0: # Is an edge
			#density = (edge_dense[y + int_r, x + int_r] + edge_dense[y - int_r, x + int_r] +
			#	edge_dense[y + int_r, x - int_r] - edge_dense[y - int_r, x - int_r])
			density = np.sum(img_src[y-sobel_eps:y+sobel_eps+1, x-sobel_eps:x+sobel_eps+1])
			if density > mx:
				mx = density
			if density < mn:
				mn = density
	print(mn, mx)
	
	for y, x in coords: # Harris corners
		if 20 < y < h - 20 and 20 < x < w - 20:
			min_dist = h
			min_y = 0
			min_x = 0
			local_list = []
			for j in range(len(points)):
				if dist((y, x), points[j]) < max_r * 1.6:
					local_list.append(j)
			for yd in range(y - int_r, y + int_r + 1):
				for xd in range(x - int_r, x + int_r + 1):
					if edge_mask[yd, xd]:
						if dist((y, x), (yd, xd)) < min_dist:
							min_dist = dist((y, x), (yd, xd))
							min_y, min_x = yd, xd
			
			edge_mask[y, x] = 1
			pure_edge.add((y, x))
			img[y, x] = np.array([1, 0, 0])
			img[y-1:y+2, x-1:x+2] = np.array([1, 0, 0])
			
			points.append((y, x))
			edge_chain.append(-1)
			edge_coors[(y, x)] = len(points) - 1
			'''for j in local_list:
				i = len(points) - 1
				if dist(points[i], points[j]) <= max_r * 2:
					draw(img, points[i], points[j])
					if not i in graph:
						graph[i] = []
					graph[i].append(j)
					if not j in graph:
						graph[j] = []
					graph[j].append(i)'''
	
	#plt.imshow(img)
	#plt.show()
	
	end = False
	first = False
	for iters in range(2):
		print("Iteration", iters)
		for p in edge_list:
			y, x = p
			#density = np.sum(img_src[y-sobel_eps:y+sobel_eps+1, x-sobel_eps:x+sobel_eps+1])
			#img[y, x] = np.array([0, 0, ((density - mn) / (mx - mn)) ** 2])
			if img_src[y, x] != 0: # Is an edge
				if random() < 0.5 or iters > 0:
					min_dist = h
					min_y = 0
					min_x = 0
					local_list = []
					for j in range(len(points)):
						if dist((y, x), points[j]) < max_r * 1.6:
							local_list.append(j)
					for yd in range(y - int_r, y + int_r + 1):
						for xd in range(x - int_r, x + int_r + 1):
							if edge_mask[yd, xd]:
								if dist((y, x), (yd, xd)) < min_dist:
									min_dist = dist((y, x), (yd, xd))
									min_y, min_x = yd, xd
					eps_loc = min(8, max(4, eps * (1 - precision[y, x]) ** 0.8)) # max(2, eps * (1 - precision[y, x]))
					max_loc = eps_loc * 1.3
					if eps_loc <= min_dist:
						edge_mask[y, x] = 1
						pure_edge.add((y, x))
						img[y, x] = np.array([1, 0, 0])
						img[y-1:y+2, x-1:x+2] = np.array([1, 0, 0])
						
						points.append((y, x))
						edge_chain.append(-1)
						edge_coors[(y, x)] = len(points) - 1
						
						'''for j in local_list:
							i = len(points) - 1
							if dist(points[i], points[j]) <= max_r * 3:
								draw(img, points[i], points[j])
								if not i in graph:
									graph[i] = []
								graph[i].append(j)
								if not j in graph:
									graph[j] = []
								graph[j].append(i)'''
		#plt.imshow(img)
		#plt.show()
	#plt.imshow(img)
	#plt.show()
	
	for iters in range(1):
		print("Iteration", iters)
		for y, x in inside_list:
			if random() < 0.5 or iters > 0:
				min_dist = h
				for yd in range(y - int_r, y + int_r + 1):
					for xd in range(x - int_r, x + int_r + 1):
						if edge_mask[yd, xd]:
							min_dist = min(min_dist, dist((y, x), (yd, xd)))
				if max(6, eps * (1 - precision[y, x]) ** 0.8) <= min_dist <= max_r:
					edge_mask[y, x] = 1
					img[y, x] = np.array([1, 0, 0])
					img[y-1:y+2, x-1:x+2] = np.array([1, 0, 0])
					
					points.append((y, x))
	
	pure_list = []
	internal_list = []
	for i in range(len(points)):
		if points[i] in pure_edge:
			pure_list.append(i)
		else:
			internal_list.append(i)
	
	for ord_list in [pure_list, internal_list]:
		for dist_k in [0.3, 0.5, 1, 1.6, 2]:
			print("CONNECT WITH K =", dist_k)
			for i in ord_list:
				dist_loc = dist_k
				y, x = points[i]
				if points[i] in pure_edge:
					#dist_loc = dist_k * max(2, eps * (1 - precision[y, x]) ** 0.8) * 2 / max_r
					dist_loc = 1.6 * min(8, max(4, eps * (1 - precision[y, x]) ** 1.3)) * 1.8 / max_r
				cprint = 260 < x < 288 and 366 < y < 395
				cprint = False
				if cprint:
					print("POINT", points[i])
					print()

				local_list = []
				local_idx = []
				
				
				#eps_loc = max(2, (eps * (1 - precision[y, x]))**0.5)
				#max_loc = eps * 1.3
				max_loc = max_r
				for j in range(len(points)):
					if dist(points[i], points[j]) <= max_loc * 4:
						local_list.append(points[j])
						local_idx.append(j)
				
				incid_list = []
				for a in range(len(local_list)):
					for b in range(len(local_list)):
						if local_idx[a] in graph and local_idx[b] in graph[local_idx[a]]:
							incid_list.append((local_idx[a], local_idx[b]))
				
				for j in local_idx:
					if points[i] in pure_edge:
						if not points[j] in pure_edge:
							continue
					if i != j and dist(points[i], points[j]) < max_loc * dist_loc:
						if cprint:
							print("TRY CONNECT TO", points[j])
						can_draw = True
						xa, xb = 0, 0
						for a, b in incid_list:
							#print(local_list[a], local_list[b])
							if intersect(points[i], points[j], points[a], points[b]):
								can_draw = False
								xa, xb = a, b
								break
						y_mean = (points[i][0] + points[j][0]) // 2
						x_mean = (points[i][1] + points[j][1]) // 2
						if img_bw[y_mean, x_mean] > 0.9:
							can_draw = False
						
						if can_draw:
							draw(img, points[i], points[j])
							if not i in graph:
								graph[i] = []
							graph[i].append(j)
							if not j in graph:
								graph[j] = []
							graph[j].append(i)
							
							incid_list.append((i, j))
						else:
							pass
							if cprint:
								print("CANNOT BECAUSE OF", points[xa], "-", points[xb])
				
				if cprint:
					img[y-3:y+4, x-3:x+4] = np.array([1, 1 / dist_k, 0])
					plt.imshow(img[366:395, 260:288])
					plt.show()
			#plt.imshow(img)
			#plt.show()

	print("Triangles constructed")
	
	ouf = open(out_list, "w")
	triangles = set()
	for i in range(len(points)):
		if i in graph:
			for j in graph[i]:
				if j in graph:
					for x in graph[i]:
						if x in graph[j]:
							tri = tuple(sorted([i, j, x]))
							if not tri in triangles:
								if not small_tri(points[i], points[j], points[x]):
									#print((i, j, x))
									triangles.add(tri)
	
	edge_outline = set()
	for i in range(len(points)):
		if points[i] in pure_edge:
			if i in graph:
				for j in graph[i]:
					if points[j] in pure_edge:
						seg = tuple(sorted([i, j]))
						if not seg in edge_outline:
							#print(seg)
							edge_outline.add(seg)
							#draw(img_bw, points[i], points[j])
	
	ouf.write(str(len(triangles)) + '\n')
	for tri in triangles:
		a = points[tri[0]]
		ouf.write(str(a[1] / w - 0.5) + ' ' + str(1 - a[0] / h - 0.5) + ' 0 ')
		a = points[tri[1]]
		ouf.write(str(a[1] / w - 0.5) + ' ' + str(1 - a[0] / h - 0.5) + ' 0 ')
		a = points[tri[2]]
		ouf.write(str(a[1] / w - 0.5) + ' ' + str(1 - a[0] / h - 0.5) + ' 0\n')
	
	ouf.write(str(len(edge_outline)) + '\n')
	for seg in edge_outline:
		a = points[seg[0]]
		ouf.write(str(a[1] / w - 0.5) + ' ' + str(1 - a[0] / h - 0.5) + ' 0 ')
		a = points[seg[1]]
		ouf.write(str(a[1] / w - 0.5) + ' ' + str(1 - a[0] / h - 0.5) + ' 0\n')
		
	ouf.close()
	#draw(img, (10, 10), (100, 100))
	
	
	print(np.sum(edge_mask))
	io.imsave(out_img, (np.clip(img, 0, 1)*255).astype(np.uint8))

process('letters/letter-0.png', 'letters/letter-0-img.png', 'letters/letter-0-list.in')
process('letters/letter-1.png', 'letters/letter-1-img.png', 'letters/letter-1-list.in')
process('letters/letter-2.png', 'letters/letter-2-img.png', 'letters/letter-2-list.in')
process('letters/letter-3.png', 'letters/letter-3-img.png', 'letters/letter-3-list.in')
process('letters/letter-4.png', 'letters/letter-4-img.png', 'letters/letter-4-list.in')
process('letters/letter-5.png', 'letters/letter-5-img.png', 'letters/letter-5-list.in')
