import sys
import getopt
import cv2
import numpy as np
from matplotlib import pyplot as plt

class MorphologicalOperators:
	def __init__(self, img):
		"""Create threshold image"""
		ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
		self.background = 0
		self.labeled_image = LabeledImage(thresh, self.background)
		self.rows,self.cols = self.labeled_image.shape()
	
	def erosion(self, se):
		print "erosion operation"

	def dilation(self, se):
		print "dilation operation"

	def opening(self, se):
		print "opening operation"

	def closing(self, se):
		print "closing operation"

	def boundary(self):
		print "boundary operation"						

	def plot(self):
		self.labeled_image.plot()

	def save(self, output_file):
		self.labeled_image.save(output_file)	
	

class LabeledImage:
	def __init__(self, matrix, background):
		"""Store matrix"""
		self.matrix = matrix
		self.background = background

	def shape(self):
		return self.matrix.shape	

	def get_pixel(self, row, col):
		"""Return a specific pixel"""
		return Pixel(self.matrix.item(row,col), row, col)

	def label_pixel(self, pixel):
		"""Label a specific pixel"""
		self.matrix.itemset((pixel.row,pixel.col), pixel.label)

	def get_neighbors(self, row, col):
		"""Return left and upper pixel"""
		if row <= 0:
			left = Pixel(self.background)
		else:
			left = self.get_pixel(row,col-1)
		if col <= 0:
			upper = Pixel(self.background)
		else:
			upper = self.get_pixel(row-1,col)
		return (left, upper)

	def get_surrounding(self, row, col):
		"""Return eight surrounding pixels"""
		locations = [[row+1,col],[row+1,col-1],[row+1,col+1],
								 [row-1,col],[row-1,col-1],[row-1,col+1],
								 [row,col-1],[row,col+1]]
		surr = []
		for loc in locations:
			row_temp = loc[0]
			col_temp = loc[1]
			row_max,col_max = self.shape()
			if row_temp >= 0 and col_temp >= 0 and row_temp < row_max and col_temp < col_max:
				px = self.get_pixel(row_temp,col_temp)
				surr.append(px)
		return surr		

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
						
def main():
	def usage():
		print 'python morphological_operators.py -i <input_file> [-o <output_file>]'

	input_file = None
	output_file = None

	"""Process command line inputs"""
	try:
		opts, args = getopt.getopt(sys.argv[1:], "hi:o:", ["help", "input=", "output="])
	except getopt.GetoptError:
		usage()
		sys.exit(2)
	for opt, arg in opts:
		if opt in ('-h', '--help'):
			usage()
			sys.exit()	
		elif opt in ("-i", "--input"):
			input_file = arg
		elif opt in ("-o", "--output"):
			output_file = arg

	if not input_file:
		usage()
		sys.exit()

	img = cv2.imread(input_file,0)
	mo = MorphologicalOperators(img)
	
	if output_file:
		mo.save(output_file)
	else:
		mo.plot()

if __name__ == "__main__":
	main()
		