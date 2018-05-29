# @cgiarrusso
import os
import csv
import numpy as np 
import pickle 
import random

# These are the parameters that you can change
# skull_set is a list of the various skulls by file name
# use_phase determines whether or not it uses phase or amplitude for the featureset (i.e. use_phase=True uses phase, use_phase=False uses amplitude)
# num_buckets is the number of buckets the program sorts into (i.e. if this were image recognition num_buckets would be the number of different image types)
# amp_buckets is a list with the ranges of the buckets
skull_set = [43215, 43214, 43203]
one_hot = False
use_phase = False
num_buckets = 1
amp_buckets = [0, 0.05382, 0.06636, 0.07801, 0.08921, 0.09957, 0.10979, 0.11991, 0.13095, 0.14261, 0.15813, 0.18544, 0.3305]

# These are the global parameters the program needs to keep stored so that it knows where it was when it is reopened
initialized = False
index_used = [0,0]
test_data = []
train_data = []

# This function reads the file containing the element number, phase, and amplitude and outputs it
def ReadFile(FileName, delimiter=","):
	assert os.path.exists(FileName)

	for l in csv.reader(open(FileName), delimiter=delimiter):
			yield float(l[0]), float(l[1]), complex(l[2])

# This function takes the data saved from the creation of the splines (skull profiles) and unpacks them from the pickles they were saved in
def unpickle_it(pickleFile):
	pickle_in = open("CNN_linear_data/skullSpline/%s.pickle" % (pickleFile),"rb") 
	spline = pickle.load(pickle_in)
	pickle_in.close()
	return spline

# This is the function that is called by the neural net and it returns training or test data
def get_data(data_set, batch_size, image_length):
	global skull_set, index_used, test_data, train_data, initialized, use_phase, amp_buckets, one_hot

	# This loads all the data the program needs
	if (initialized == False):
		skull_data = []
		image_data = []
		feature_set_data = []
		all_data = []

		# Pulling the data from storage
		for i in range(len(skull_set)):
			skull_data.append(unpickle_it(skull_set[i]))
			for j in ReadFile('CNN_linear_data/elementData/%s.csv' % (skull_set[i])):
				if use_phase:
					feature_set_data.append(j[1])
				else:
					j_abs = np.absolute(j[2])
					feature_set_data.append(j_abs)
		
		# Modifying the data so that it can be read easily
		skull_data = list(skull_data)
		for i in skull_data:
			for j in i:
				image_data.append(j)
		image_data = list(image_data)
		feature_set_data = list(feature_set_data)

		# Merging the data into one giant tuple 
		for i in range(len(image_data)):
			all_data.append([image_data[i],feature_set_data[i]]) 
		random.shuffle(all_data)

		# Splitting the data into a training set and testing set
		all_data = np.array(all_data)
		test_ratio = int(len(all_data)*0.9)
		train_data = all_data[:test_ratio,:]
		test_data = all_data[test_ratio:,:]

		# Recording the feature set to check for bugs
		with open('CNN_linear_data/ampcheck.csv', "w") as output:		
			writer = csv.writer(output, lineterminator='\n')		
			for val in feature_set_data:		
				writer.writerow([val]) 

		# Marking the task as finished so it doesn't do it again
		initialized = True
	
	# Using the correct data set
	test_or_train = 0
	if (data_set == 'test'):
		test_or_train = 1
		data_used = test_data
	else:
		data_used = train_data

	# Shaping the output so that it matches what the neural net needs
	image_data = np.zeros((batch_size, image_length))
	feature_set_data = np.zeros((batch_size, num_buckets))

	# Making sure to start where the program left off at so as to go through all the data before recycling any
	begin = index_used[test_or_train]
	end = index_used[test_or_train] + batch_size
	if end >= len(data_used):
		index_used[test_or_train] = end - len(data_used)

	# Loading the data
	index = 0
	for j in range(begin, end):
		i = j
		if j >= len(data_used):
			i = j - len(data_used)
		image_data[index] = data_used[i,0]
		
		if one_hot:
			# Here it creates the one-hot encoded vector based on the number of buckets
			if use_phase:
				feature_set_data[index][int((data_used[i,1]+3.2)*5)] = 1
			else:
				for k in range(len(amp_buckets)-1):
					if data_used[i,1] >= amp_buckets[k]:
						if data_used[i,1] <= amp_buckets[k+1]:
							feature_set_data[index][k] = 1
							if k == 0:
								feature_set_data[index][len(amp_buckets)-2] = 1
								feature_set_data[index][k+1] = 1
							if k == len(amp_buckets)-2:
								feature_set_data[index][k-1] = 1
								feature_set_data[index][0] = 1
							else:
								feature_set_data[index][k-1] = 1
								feature_set_data[index][k+1] = 1
		else:
			feature_set_data[index][0]=data_used[i,1]+3.14159265
		index += 1

	# Saving the last index it left off at
	index_used[test_or_train] += batch_size

	return image_data, feature_set_data


# The following is a function I'm currently working on that would just take the profile of the skull and cut away all the noise
# {______----/\/\------}
# Above is the profile of the skull, and function would cut out everything except the "/\/\" which is the skull
# 
# 
# greatestval = 0
# def edit_spline(spline):
# 	global greatestval
# 	newspline = []
# 	for i in range(len(spline)-10):
# 		avg = 0
# 		for j in range(10):
# 			avg += int(spline[i+j])
# 		if (avg >= 1000):
# 			newspline.append(spline[i])
# 	if (greatestval < len(newspline)):
# 		greatestval = len(newspline)
# 	return newspline
