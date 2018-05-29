# @cgiarrusso
import os
import csv
import numpy as np
import pickle

skull_set = [43215, 43214, 43203]

# This function takes the data saved from the creation of the splines (skull profiles) and unpacks them from the pickles they were saved in
def unpickle_it(pickleFile):
	pickle_in = open("CNN_linear_data/skullSpline/%s.pickle" % (pickleFile),"rb") 
	spline = pickle.load(pickle_in)
	pickle_in.close()
	return spline

# [1000, 1] spline of the skull as input
# Range -1000 -> 100 background noise, 100 -> 4000+ skull
def classifier(spline):
	scope = [1, 10]
	activation = [300, 200]
	one_or_two = 0

	length = 0
	max1 = 0
	max2 = 0
	low = 0

	for i in range(len(spline)-10):
		average = 0
		for j in range(scope[one_or_two]):
			average += spline[i+j]
		if average >= activation[one_or_two]*scope[one_or_two]:
			one_or_two = 1
			length += 1
			if low == max1:
				if spline[i] > max1:
					max1 = spline[i]
					low = max1
				if spline[i] < max1:
					if spline[i] < low:
						low = spline[i]
			else:
				if spline[i] < low:
					if max2 == 0:
						low = spline[i]
				else:
					if spline[i] > max2:
						max2 = spline[i]
	return length, max1, max2, low

def main():

	global skull_set

	skull_data = []
	image_data = []
	spline_data = []

	# Pulling the data from storage
	for i in range(len(skull_set)):
		skull_data.append(unpickle_it(skull_set[i]))
		
	# Modifying the data so that it can be read easily
	skull_data = list(skull_data)
	for i in skull_data:
		for j in i:
			image_data.append(j)
	image_data = list(image_data)

	for i in image_data:
		spline_data.append(classifier(i))
	spline_data = list(spline_data)

	# Recording the feature set to check for bugs
	with open('CNN_linear_data/splinecheck.csv', "w") as output:		
		writer = csv.writer(output, lineterminator='\n')		
		for val in spline_data:		
			writer.writerow([val]) 

if __name__ == "__main__":
	main()









