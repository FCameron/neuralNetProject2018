# @cgiarrusso
import os
import csv
import numpy as np 
import scipy.ndimage as ndimage
import scipy.io as io
import pickle 

# These are the parameters that you can change
# skullN is the skull you are getting the spline of and is the index skullSet
# skullSet is the list of all the skulls
skullN = 0
skullSet = [43215, 43214, 43203]

# skullMap is loading the CT of the skulls from the matlab file
# imdim is the number of layers deep the CT is
skullMap = io.loadmat('CNN_linear_data/skullCT/%s.mat' % (skullSet[skullN]))['ctSkull']
imdim = [304, 322, 271]

# Instead of using real world corrdinates, I'm using the CT coordinates (i.e. 0,0,z is one corner and 512,512,z is another)
# This program transforms the real world coordinates to CT coordinates
def coordinate_mapping(x, y, z):
	global xTarg, yTarg, zTarg, skullN
	if (skullN == 0):
		xCoeff, yCoeff, zCoeff = 0.4531, 0.4531, 0.6250
		xShift, yShift, zShift = 116.2266, 107.9266, 9.8125
		xTargs, yTargs, zTargs = 13.9176, 3.2211, 67.7672
	if (skullN == 1):
		xCoeff, yCoeff, zCoeff = 0.5059, 0.5059, 0.6250
		xShift, yShift, zShift = 130.7529, 124.0529, 30.3125
		xTargs, yTargs, zTargs = 12.6627, 22.8653, 54.7873
	if (skullN == 2):
		xCoeff, yCoeff, zCoeff = 0.4766, 0.4766, 0.6250
		xShift, yShift, zShift = 122.2383, 122.2383, 16.3125
		xTargs, yTargs, zTargs = -7.2121, 0.6630, 60.7222

	xTarg, yTarg, zTarg = (xTargs+xShift)/xCoeff, (yTargs+yShift)/yCoeff, (zTargs+zShift)/zCoeff
	return (x+xShift)/xCoeff, (y+yShift)/yCoeff, (z+zShift)/zCoeff 

# This program takes the CT of the skull and the locations of the elements and makes a spline, finding the profile of the skull
def interpolation(xElem, yElem, zElem):
	global skullMap, imdim, skullN, xTarg, yTarg, zTarg

	# This creates a 3D map where each point has the HU at that point from the skull CT
	x, y, z = np.mgrid[0:512:1, 0:512:1, 0:imdim[skullN]:1]
	skull = skullMap[x, y, z]

	# This creates the line from the focus to the element
	x, y, z = np.linspace(xElem, xTarg, 1000),\
			  np.linspace(yElem, yTarg, 1000),\
			  np.linspace(zElem, zTarg, 1000)

	# This is what actually takes the spline, mapping the best value it can to each of the 1000 values on the line
	spline = ndimage.map_coordinates(skull, np.vstack((x,y,z)))

	for i in range(len(spline)):
		if spline[i] < -100:
			spline[i] = 0

	return spline


# This program saves the splines so that I can use them
def pickle_it(spline, pickleFile, numberSkull):
	pickle_out = open("CNN_linear_data/skullSpline/%s.pickle" % (pickleFile[numberSkull]),"wb")
	pickle.dump(spline,pickle_out)
	pickle_out.close()

# This is the main function that I'd call on to create all the splines
def splining(skull):
	global skullMap, skullSet, skullN
	
	skullN = skull

	spline = []
	counter = 0
	for (xCoords, yCoords, zCoords) in ReadFile('CNN_linear_data/transducerCoordinates/%s.csv' % (skullSet[skullN])):
		x, y, z = coordinate_mapping(xCoords,yCoords,zCoords)
		spline.append(interpolation(x, y, z))
		counter += 1
		print(counter)

	pickle_it(spline, skullSet, skullN)

# This function reads csv files
def ReadFile(FileName, delimiter=","):
	assert os.path.exists(FileName)

	for l in csv.reader(open(FileName), delimiter=delimiter):
		yield float(l[0]), float(l[1]), float(l[2])

for i in range(3):
	splining(i)