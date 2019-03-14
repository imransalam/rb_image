"""
Scikit Image wrapper for RB utility functions
"""
from skimage import io, feature, measure, draw, transform
import numpy as np
from collections import defaultdict

def show(img):
  '''
  Inputs: Accepts either a list of Numpy arrays objects or a single Numpy array object
  Outputs: None
  Example:
  >> img = np.ones((10,10))
  >> show(img)
  >> imgs = [img, np.random.rand(10,10)]
  >> show(imgs)
  '''
  if isinstance(img, list):
  	io.imshow_collection(img)
  else:
  	io.imshow(img)
  	io.show()

def draw_lines( lines, sz = [640,640] ):
	'''
	Inputs: Accepts list of lines, a line is a tuple, which contains two tuples, 
	point A and point B. A size sz which is default to (640,640)
  	Outputs: A Boolean Image of Lines,
	Example:
	>> lines = [
		((10,10),(20,20)),
		((30,30),(50,50))
	]
	>> draw_lines(lines, sz = (100,100))
	'''
	img = np.zeros(sz)
	for l in lines:
		p1, p2 = l
		r0, c0 = p1
		r1, c1 = p2
		rr, cc, val = draw.line_aa( int(c0), int(r0), int(c1), int(r1) )
		img[rr, cc] = 1
	return img.astype(bool)

def invert_xy(pols):
	'''
	Inputs: Accepts list of polygons
  	Outputs: Lists of polygons with their Xs and Ys inverted
	Example:
	>> POLS = [
					[[404, 236], [496, 357], [304, 492]], 
					[[125, 270], [157, 307], [131, 323]], 
			]
	>> invert_xy(POLS)
	'''
	new_pols = []
	for i in range(len(pols)):
		temp_pol = []
		for j in range(len(pols[i])):
			temp_pol.append([pols[i][j][1], pols[i][j][0]])
		new_pols.append(temp_pol)
	return new_pols

def pols_to_img(pols, sz = [640,640]):
	'''
	Inputs: Accepts list of polygons and a size tuple/list for output image.
	Outputs: Integer Image of polygons drawn
	Example:
	>> POLS = [
					[[404, 236], [496, 357], [304, 492]], 
					[[125, 270], [157, 307], [131, 323]], 
			]
	>> pols_to_img(POLS, sz=[640,640])
	'''
	img = np.zeros(sz, dtype=int)
	for ii,pol in enumerate(pols):
		if pol:
			rr, cc = draw.polygon( [ pol[i][0] for i in range(len(pol)) ], [ pol[i][1] for i in range(len(pol)) ] )
			img[ rr, cc ] = ii+1
	return img

def draw_pols_as_lines(pols, sz = [640,640]):
	'''
	Inputs: Accepts list of polygons and a size tuple/list for output image.
	Outputs: Boolean Image of polygons lines drawn
	Example:
	>> POLS = [
					[[404, 236], [496, 357], [304, 492]], 
					[[125, 270], [157, 307], [131, 323]], 
			]
	>> draw_pols_as_lines(POLS, sz=[640,640])
	'''
	img = np.zeros(sz)
	for pol in pols:
		if pol:
			for i in range(len(pol)):
				p0, p1 = pol[i], pol[(i+1)%len(pol)]
				rr, cc, val = draw.line_aa(int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1]))
				img[rr, cc] = 1
	return img.astype(bool)

def select_region_pol(ri):
	section_pols = measure.find_contours(ri, 0.1)
	selected_pol = section_pols[0]
	max_pol_size = -1
	for sp in section_pols:
		rr, cc = draw.polygon([spi[0] for spi in sp], [spi[1] for spi in sp])
		if len(rr) > max_pol_size:
			max_pol_size = len(rr)
			selected_pol = sp
	return selected_pol

def img_to_pols_simple(img, tol_contour=0.1, tol_approximation=5):
	'''
	Inputs: Accepts 2D image and a tolerance factor tol=0.1
	Outputs: Polygons of the objects in Image
	Example:
	>> img = np.zeros((100,100))
	>> img[25:35,25:35] = 1
	>> img[55:65,55:65] = 1
	>> img_to_pols_simple(img, tol=0.1)
	'''
	im2 = measure.find_contours(img, level=tol_contour)
	im2 = [measure.approximate_polygon(i, tol_approximation).tolist() for i in im2]

	for im in range(len(im2)):
		for i in range(len(im2[im])):
			im2[im][i] = [int(im2[im][i][0]), int(im2[im][i][1])]
	return im2

def fix_points(sections_pols, sz=[640,640], delta=20):
	'''
	Inputs: Accepts list of polygons, a size tuple/list for output image and delta for range.
	Default is sz=[640,640], delta=20
	Outputs: Boolean Image of polygons lines drawn
	polygons ko chipkata hai based on distance (delta)
	Example:
	>> POLS = [
					[[404, 236], [496, 357], [304, 492]], 
					[[125, 270], [157, 307], [131, 323]], 
			]
	>> fix_points(sections_pols)
	'''
	img = np.zeros(sz)
	for sp in sections_pols:
		for p in sp:
			p0 = int(p[0])
			p1 = int(p[1])
			img[p0 - delta:p0 + delta, p1 - delta:p1 + delta] = 1
	img = measure.label(img)
	point_to_region = defaultdict(lambda: [])
	for i in range(len(sections_pols)):
		for j in range(len(sections_pols[i])):
			p = sections_pols[i][j]
			r = img[int(p[0]), int(p[1])]
			point_to_region[r].append((p, i, j))
	sections_pols_1 = copy.deepcopy(sections_pols)
	for r in point_to_region:
		points = [rps[0] for rps in point_to_region[r]]
		mean_point = np.mean(points, 0)
		for rps in point_to_region[r]:
			pt, pol_idx, pt_idx = rps
			sections_pols_1[pol_idx][pt_idx] = mean_point
	return sections_pols_1, img

def simplify_pol(pol, tol=2, max_iter=20):
	'''
	Inputs: Accepts a polygon, tolerance=2 and maximum iteration.
	Default values are tol=2 and max_iter=20
	Outputs: Simplified Polygon. Works kind of like measure.approximate_polygon.
	Example:
	>> POL = [
					[125, 270], [157, 307], [131, 323] 
			]
	>> simplify_pol(POL)
	'''
	iiter = 0
	while True:
		iiter = iiter + 1
		pol1 = []
		can_skip = set()
		i = 1
		while i < len(pol) - 1:
			p0, p1 = pol[i - 1], pol[i + 1]
			rr, cc, val = draw.line_aa(int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1]))
			pts = set(zip(rr, cc))
			pi = set()
			for x in range(int(int(pol[i][0]) - tol / 2), int(int(pol[i][0]) + tol / 2) + 1):
				for y in range(int(int(pol[i][1]) - tol / 2), int(int(pol[i][1]) + tol / 2) + 1):
					pi.add((x, y))
			if len(pi.intersection(pts)) > 0:
				can_skip.add(i)
				i = i + 1
			i = i + 1
		if len(can_skip) == 0 or iiter >= max_iter:
			break
		for i in range(len(pol)):
			if not i in can_skip:
				pol1.append(pol[i])
		pol = pol1
	return pol

def get_region_polygons(roof_regions):
	'''
	Inputs: Accepts 2D image of label nature
	Outputs: Polygons of the objects in Image
	Example:
	>> img = np.zeros((100,100))
	>> img[25:35,25:35] = 1
	>> img[55:65,55:65] = 2
	>> get_region_polygons(img)
	'''
	regionimg = [np.array((roof_regions == i)) for i in np.unique(roof_regions)[1:]]
	pols = [measure.approximate_polygon(select_region_pol(ri), 3).astype(int).tolist() for ri in regionimg]
	pols = [simplify_pol(pi) for pi in pols]
	return pols

def angle_between_new(p1, p2):
	'''
	Inputs: Accepts two Points of line
	Outputs: Angle in Degrees in 2D plane
	Example:
	>> p1,p2 = (4,6),(12,18)
	>> angle_between_new(p1,p2)
	'''
	if p2[0] < p1[0]:
		pt = p2
		p2 = p1
		p1 = pt
	b = abs(p2[1] - p1[1])
	ba = p2[1] - p1[1]
	h = np.sqrt(np.power(p1[0] - p2[0], 2) + np.power(p1[1] - p2[1], 2))
	if abs(h) < 0.01:
		return 0.0
	else:
		if ba > 0:
				return (180 - (np.arccos(1.0 * b / h) * 180.0 / np.pi % 360.0))
		return np.arccos(1.0 * b / h) * 180.0 / np.pi % 360.0

def getRotationAngle(img):
	'''
	Inputs: Accepts a binary Image
	Outputs: Rotation angle in Degrees
	Example:
	>> img = np.zeros((10,10))
	>> img[4:6,4:6] = 1
	>> getRotationAngle(img)
	'''
	edges = feature.canny(img)
	lines = transform.probabilistic_hough_line(edges)
	angle = 0
	max_len = -1
	for l in lines:
		a, b = l
		d = np.power(a[0] - b[0], 2) + np.power(a[1] - b[1], 2)
		if d > max_len:
			max_len = d
			angle = angle_between_new(a, b)
	return angle
