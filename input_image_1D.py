import os
import csv
import numpy as np 
import scipy as sp
import scipy.ndimage as ndimage
import scipy.io as io
import pickle 
import random

skull_set = [43215, 43214, 43203]
initialized = False

index_used = [0,0]
x_data = []
y_data = []
all_data = []
test_data = []
train_data = []
num_classes = 63

def unpickle_it(pickleFile):
	pickle_in = open("CNN_linear_data/skullSpline/skullSpline%s.pickle" % (pickleFile),"rb") 
	spline = pickle.load(pickle_in)
	pickle_in.close()
	return spline

def get_data(data_set, batch_size, tsvlocation=""):
	global skull_set, index_used, y_data, x_data, all_data, test_data, train_data, initialized


	if (initialized == False):
		for i in range(len(skull_set)):
			x_data.append(unpickle_it(skull_set[i]))
			for y_ in ReadFile('CNN_linear_data/phaseChange/phaseChange%s.csv' % (skull_set[i]), 2):
				y_data.append(y_[1])
		x_data = list(x_data)
		y_data = list(y_data)
		for i in range(len(x_data)):
			for j in range(len(x_data[i])):
				all_data.append([x_data[i][j][:],y_data[j]])
				# for k in range(30):
				# 	all_data.append([np.roll(x_data[i][j][:],k*10),y_data[j]])
				# 	all_data.append([np.flip(np.roll(x_data[i][j][:],k*10),0), y_data[j]])
				# all_data.append([np.flip(x_data[i][j][:],0), y_data[j]])
		random.shuffle(all_data)
		all_data = np.array(all_data)
		test_ratio = int(len(all_data)*0.9)
		train_data = all_data[:test_ratio,:]
		test_data = all_data[test_ratio:,:]
		initialized = True
	
	test_or_train = 0
	if (data_set == 'test'):
		test_or_train = 1
		data_used = test_data
	else:
		data_used = train_data

	x_data_ = np.zeros((batch_size, 1000))
	y_data_ = np.zeros((batch_size, num_classes))
	index = 0

	begin = index_used[test_or_train]
	end = index_used[test_or_train] + batch_size

	if end >= len(data_used):
		index_used[test_or_train] = end - len(data_used)

	for j in range(begin, end):
		i = j
		if j >= len(data_used):
			i = j - len(data_used)
		x_data_[index] = data_used[i,0]
		y_data_[index][int((data_used[i,1]+3)*10)] = 1
		index += 1

	index_used[test_or_train] += batch_size

	return x_data_, y_data_

def ReadFile(FileName, Outputs, delimiter=","):
	assert os.path.exists(FileName)

	for l in csv.reader(open(FileName), delimiter=delimiter):
		if Outputs == 2:
			yield float(l[0]), float(l[1])
		else:
			yield float(l[0]), float(l[1]), float(l[2])