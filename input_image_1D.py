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
num_classes = 32
greatestval = 0


def edit_spline(spline):
	global greatestval
	newspline = []
	for i in range(len(spline)-10):
		avg = 0
		for j in range(10):
			avg += int(spline[i+j])
		if (avg >= 1000):
			newspline.append(spline[i])
	if (greatestval < len(newspline)):
		greatestval = len(newspline)
	return newspline


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
			for y_ in ReadFile('CNN_linear_data/phaseChange/phaseChange%s.csv' % (skull_set[i]), 'phase'):
			# for y_ in ReadFile('CNN_linear_data/ampChange/ampChange%s.csv' % (skull_set[i]), 'amp'):
				# y_abs = np.absolute(y_[0])
				# y_data.append(y_abs)
				y_data.append(y_[1])
		x_data = list(x_data)
		y_data = list(y_data)
		x_data2 = []
		for k in x_data:
			for i in k:
				# x_data2.append(edit_spline(i))
				x_data2.append(i)
		x_data2 = list(x_data2)
		# x_data3 = x_data2
		# for i in range(len(x_data3)):
		# 	x_data2[i] = np.zeros(244)
		# 	for j in range(len(x_data3[i])):
		# 		x_data2[i][j] = x_data3[i][j]
		# x_data3 = []
		x_data = x_data2
		x_data2 = []
		# with open('CNN_linear_data/ampChange/ampcheck.csv', "w") as output:
		# 	writer = csv.writer(output, lineterminator='\n')
		# 	for val in x_data2:
		# 		writer.writerow([val]) 
		for i in range(len(x_data)):
			all_data.append([x_data[i],y_data[i]]) 
		# for i in range(len(x_data)):
		# 	for j in range(len(x_data[i])):
		# 		all_data.append([x_data[i][j][:],y_data[j]])
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
		y_data_[index][int((data_used[i,1]+3.2)*5)] = 1
		# y_data_[index][int((data_used[i,1]/.01))] = 1
		index += 1

	index_used[test_or_train] += batch_size

	return x_data_, y_data_

def ReadFile(FileName, Outputs, delimiter=","):
	assert os.path.exists(FileName)

	for l in csv.reader(open(FileName), delimiter=delimiter):
		if Outputs == 'phase':
			yield float(l[0]), float(l[1])
		elif Outputs == 'amp':
			yield complex(l[2]), float(l[0])
		else:
			yield float(l[0]), float(l[1]), float(l[2])








