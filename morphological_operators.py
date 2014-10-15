import sys
import getopt
import cv2
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy

class MorphologicalOperators:
	def __init__(self, img, se):
		"""Create threshold image"""
		self.background = 0
		self.foreground = 1
		ret,thresh = cv2.threshold(img, 127, self.foreground, cv2.THRESH_BINARY)
		self.original_image = LabeledImage(thresh, None, None, self.background)
		self.labeled_image = LabeledImage(thresh, None, None, self.background)
		self.rows,self.cols = self.labeled_image.shape()
		self.se = StructuringElement(se)
	
	def erosion(self):
		new_image = LabeledImage(None, self.rows, self.cols, self.background)
		for i in xrange(self.rows):
			for j in xrange(self.cols):
				current = self.labeled_image.get_pixel(i,j)
				new = Pixel(current.label, current.row, current.col)
				if (current.is_not_label(self.background)):
					coords = self.se.get_coords(current.row, current.col)
					neighbors = self.labeled_image.get_pixels(coords)
					other_bg_px = False
					for n in neighbors:
						if n.is_label(self.background):
							other_bg_px = True
					if other_bg_px and len(neighbors) > 0:
						new.label = self.background
				new_image.label_pixel(new)		
		self.labeled_image = deepcopy(new_image)		

	def dilation(self):
		new_image = LabeledImage(None, self.rows, self.cols, self.background)
		for i in xrange(self.rows):
			for j in xrange(self.cols):
				current = self.labeled_image.get_pixel(i,j)
				new = Pixel(current.label, current.row, current.col)
				if (current.is_label(self.background)):
					coords = self.se.get_coords(current.row, current.col)
					neighbors = self.labeled_image.get_pixels(coords)
					other_fg_px = False
					for n in neighbors:
						if n.is_not_label(self.background):
							other_fg_px = True
					if other_fg_px and len(neighbors) > 0:
						new.label = self.foreground
				new_image.label_pixel(new)
		self.labeled_image = deepcopy(new_image)	

	def opening(self):
		self.erosion()
		self.dilation()

	def closing(self):
		self.dilation()
		self.erosion()

	def boundary(self):
		self.erosion()
		boundary = np.subtract(self.original_image.matrix, self.labeled_image.matrix)
		self.labeled_image.matrix = deepcopy(boundary)
								
	def plot(self):
		self.labeled_image.plot()

	def save(self, output_file):
		self.labeled_image.save(output_file)	

	def save_text(self):
		np.savetxt('in.csv', self.original_image.matrix, delimiter=',', fmt='%d')
		np.savetxt('out.csv', self.labeled_image.matrix, delimiter=',', fmt='%d')	
	

class LabeledImage:
	def __init__(self, matrix, rows, cols, background):
		"""Store matrix"""
		if matrix is None: 
			self.matrix = np.zeros((rows,cols), dtype=np.int)
		else:	
			self.matrix = matrix
		self.background = background

	def shape(self):
		return self.matrix.shape	

	def get_pixel(self, row, col):
		"""Return a specific pixel"""
		return Pixel(self.matrix.item(row,col), row, col)

	def get_pixels(self, coords):
		pixels = []
		row_max, col_max = self.shape()
		for c in coords:
			row_temp = c[0]
			col_temp = c[1]
			if row_temp >= 0 and col_temp >= 0 and row_temp < row_max and col_temp < col_max:
				px = self.get_pixel(row_temp,col_temp)
				pixels.append(px)
		return pixels		

	def label_pixel(self, pixel):
		"""Label a specific pixel"""
		self.matrix.itemset((pixel.row,pixel.col), pixel.label)

	def plot(self):
		plt.imshow(self.matrix, interpolation = 'nearest')
		plt.xticks([]), plt.yticks([])
		plt.show()

	def save(self, output_file):
		plt.imshow(self.matrix, interpolation = 'nearest')
		plt.xticks([]), plt.yticks([])
		plt.savefig(output_file, bbox_inches='tight')		
	

class Pixel:
	def __init__(self, label, row, col):
		"""Create a pixel with label and coordinates"""
		self.label = label
		self.row = row
		self.col = col

	def is_label(self, label):
		"""Test if pixel has a label"""
		if self.label == label:
			return True
		else:
			return False

	def is_not_label(self, label):
		"""Test if pixel doesn't have a label"""
		if self.label != label:
			return True
		else:
			return False


class StructuringElement:
	def __init__(self, se):
		self.matrix = se
		self.rows,self.cols = self.matrix.shape
		if (self.rows % 2 == 0) or (self.cols % 2 == 0):
			print 'Dimensions of structuring element must be odd'
			sys.exit()
		self.origin = Pixel(None, self.rows/2, self.cols/2)

	def get_coords(self, row, col):
		coords = []
		for i in xrange(self.rows):
			for j in xrange(self.cols):
				current = self.get_pixel(i,j)
				if current.label != 0:
					row_dist = current.row - self.origin.row
					col_dist = current.col - self.origin.col
					new_row = row + row_dist
					new_col = col + col_dist
					if not(row_dist == 0 and col_dist == 0):
						coords.append([new_row,new_col])
		return coords

	def get_pixel(self, row, col):
		"""Return a specific pixel"""
		return Pixel(self.matrix.item(row,col), row, col)				
						
def main():
	def usage():
		print 'python morphological_operators.py -m <morph_op> -s <se> -i <inputf> [-o <outputf>]'

	morph_op = None
	se_in = None
	inputf = None
	outputf = None

	"""Process command line inputs"""
	try:
		opts, args = getopt.getopt(sys.argv[1:], "hm:s:i:o:", ["help", "morph_op=", "se=", 
								 "inputf=", "outputf="])
	except getopt.GetoptError:
		usage()
		sys.exit(2)
	for opt, arg in opts:
		if opt in ('-h', '--help'):
			usage()
			sys.exit()
		elif opt in ("-m", "--morph_op"):
			morph_op = arg
		elif opt in ("-s", "--se"):
			se_in = arg		
		elif opt in ("-i", "--inputf"):
			inputf = arg
		elif opt in ("-o", "--outputf"):
			outputf = arg

	if not morph_op or not se_in or not inputf:
		usage()
		sys.exit()

	if morph_op not in ("erosion", "dilation", "opening", "closing", "boundary"):
		print 'Available morphological operators are erosion, dilation, opening, closing '\
					'amd boundary'
		sys.exit()				

	if se_in == 'rect3x3':
		se = np.ones((3,3), dtype=np.int)		 		
	elif se_in == 'rect5x5':
		se = np.ones((5,5), dtype=np.int)		 		
	elif se_in == 'rect7x7':
		se = np.ones((7,7), dtype=np.int)
	elif se_in == 'rect9x9':
		se = np.ones((9,9), dtype=np.int)
	elif se_in == 'rect11x11':
		se = np.ones((11,11), dtype=np.int)
	elif se_in == 'rect13x13':
		se = np.ones((13,13), dtype=np.int)	
	else:
		print 'Available structuring elements are rect3x3, rect5x5, rect7x7, rect9x9, '\
					'rect11x11, rect13x13'
		sys.exit()			 					 		

	

	
	# good for palm
	# arr = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	# 			 [0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0],
	# 			 [0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0],
	# 			 [0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
	# 			 [0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
	# 			 [0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0],
	# 			 [0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
	# 			 [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
	# 			 [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
	# 			 [0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
	# 			 [0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
	# 			 [0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0],
	# 			 [0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0],
	# 			 [0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
	# 			 [0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0],
	# 			 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	# 			 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
	
	#good for palm
	# arr = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	# 			 [0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0],
	# 			 [0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0],
	# 			 [0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
	# 			 [0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
	# 			 [0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0],
	# 			 [0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
	# 			 [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
	# 			 [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
	# 			 [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
	# 			 [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
	# 			 [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0],
	# 			 [0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0],
	# 			 [0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0],
	# 			 [0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],
	# 			 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	# 			 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]			 
	
	# arr =[[1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
	# 			[0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
	# 			[0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
	# 			[0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
	# 			[0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1],
	# 			[0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1],
	# 			[0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1]]
	# arr = [[1,1,1,0,0],
	# 			 [0,1,1,1,0],
	# 			 [0,0,1,1,1],
	# 			 [0,0,0,1,1],
	# 			 [0,0,0,0,1]]
	# arr =[[1,1,1,1,1,1,0,0,0,0,0,0,0],
	# 			[0,1,1,1,1,1,1,0,0,0,0,0,0],
	# 			[0,0,1,1,1,1,1,1,0,0,0,0,0],
	# 			[0,0,0,1,1,1,1,1,1,0,0,0,0],
	# 			[0,0,0,0,1,1,1,1,1,1,0,0,0],
	# 			[0,0,0,0,0,1,1,1,1,1,1,0,0],
	# 			[0,0,0,0,0,0,1,1,1,1,1,1,0]]
	
	

	

	# arr =  [[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
	# 				[0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],
	# 				[0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
	# 				[0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
	# 				[0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
	# 				[0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
	# 				[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
	# 				[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
	# 				[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
	# 				[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
	# 				[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
	# 				[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
	# 				[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
	# 				[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
	# 				[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
	# 				[0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
	# 				[0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
	# 				[0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
	# 				[0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
	# 				[0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],
	# 				[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]]
				 
			
			  			 
	img = cv2.imread(inputf, 0)
	mo = MorphologicalOperators(img, se)

	if morph_op == "erosion":
		mo.erosion()
	elif morph_op == "dilation":
		mo.dilation()
	elif morph_op == "opening":
		mo.opening()
	elif morph_op == "closing":
		mo.closing()
	elif morph_op == "boundary":
		mo.boundary()				

	if outputf:
		mo.save(outputf)
	else:
		mo.plot()

if __name__ == "__main__":
	main()
		