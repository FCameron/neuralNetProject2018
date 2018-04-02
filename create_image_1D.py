import os
import csv
import numpy as np 
import scipy as sp
import scipy.ndimage as ndimage
import scipy.io as io
import pickle 
import random

skullN = 0
skullSet = [43215, 43214, 43203]
skullMap = io.loadmat('CNN_linear_data/skullCT/skullCT%s.mat' % (skullSet[skullN]))['ctSkull']

imdim = [304, 322, 271]

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

def interpolation(xElem, yElem, zElem):
	global skullMap, imdim, skullN

	global xTarg, yTarg, zTarg

	x, y, z = np.mgrid[0:512:1, 0:512:1, 0:imdim[skullN]:1]
	skull = skullMap[x, y, z]

	x, y, z = np.linspace(xElem, xTarg, 1000),\
			  np.linspace(yElem, yTarg, 1000),\
			  np.linspace(zElem, zTarg, 1000)

	return ndimage.map_coordinates(skull, np.vstack((x,y,z)))

def pickle_it(spline, pickleFile, numberSkull):
	pickle_out = open("CNN_linear_data/skullSpline/skullSpline%s.pickle" % (pickleFile[numberSkull]),"wb")
	pickle.dump(spline,pickle_out)
	pickle_out.close()


def splining(skull):
	global skullMap, skullSet, skullN
	
	skullN = skull

	spline = []
	counter = 0
	for (xCoords, yCoords, zCoords) in ReadFile('CNN_linear_data/transducerCoordinates/transducerCoordinates%s.csv' % (skullSet[skullN]), 3):
		x, y, z = coordinate_mapping(xCoords,yCoords,zCoords)
		spline.append(interpolation(x, y, z))
		counter += 1
		print(counter)

	pickle_it(spline, skullSet, skullN)

def ReadFile(FileName, Outputs, delimiter=","):
	assert os.path.exists(FileName)

	for l in csv.reader(open(FileName), delimiter=delimiter):
		if Outputs == 2:
			yield float(l[0]), float(l[1])
		else:
			yield float(l[0]), float(l[1]), float(l[2])