# @cgiarrusso
from input_image_1D import get_data
import tensorflow as tf 
import os
import os.path
import csv

length = 1000
nLabel = 1
LOGDIR = "tensorboardphase4/"
LABELS = "metadata.tsv"

def conv_layer(input, size_in, size_out, name="conv"):
	with tf.name_scope(name):
		w = tf.Variable(tf.truncated_normal([5, size_in, size_out], stddev=0.1), name="W")
		b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
		conv = tf.nn.conv1d(input, w, stride=1, padding="SAME")
		act = tf.nn.relu(conv + b)
		tf.summary.histogram("weights", w)
		tf.summary.histogram("biases", b)
		tf.summary.histogram("activations", act)
		return tf.layers.max_pooling1d(act, pool_size=2, strides=2, padding="SAME")

def fc_layer(input, size_in, size_out, name="fc"):
	with tf.name_scope(name):
		w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
		b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
		act = tf.matmul(input, w) + b
		tf.summary.histogram("weights", w)
		tf.summary.histogram("biases", b)
		tf.summary.histogram("activations", act)
		return act

def cnn_model(learning_rate, use_two_fc, use_two_conv, hparam):
	tf.reset_default_graph()
	sess=tf.Session()

	x = tf.placeholder(tf.float32, shape=[None, length], name="x")
	x_image = tf.reshape(x, [-1, length, 1])
	y = tf.placeholder(tf.float32, shape=[None, nLabel], name="labels")

	if use_two_conv:
		conv1 = conv_layer(x_image, 1, 32, "conv1")
		conv_out = conv_layer(conv1, 32, 64, "conv2")
	else:
		conv1 = conv_layer(x_image, 1, 32, "conv1")
		conv2 = conv_layer(conv1, 32, 64, "conv2")
		conv_out = conv_layer(conv2, 64, 128, "conv3")

	flattened = tf.reshape(conv_out, [-1, 250*64])

	if use_two_fc:
		fc1 = fc_layer(flattened, 250*64, 1024, "fc1")
		relu = tf.nn.relu(fc1)
		tf.summary.histogram("fc1/relu", relu)
		logits = fc_layer(fc1, 1024, nLabel, "fc2")
	else:
		logits = fc_layer(flattened, 250*64, nLabel, "fc")

	global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)

	with tf.name_scope("xent"):
		error1 = tf.abs(logits-y)
		error2 = tf.abs(logits+6.28-y)
		error3 = tf.abs(logits-6.28-y)
		xent = tf.reduce_sum(tf.pow(tf.minimum(tf.minimum(error1, error2), error3), 2), name="xent")
		# xent = tf.reduce_sum(tf.pow((logits-y), 2), name="xent")
		tf.summary.scalar("xent", xent)

	with tf.name_scope("train"):
		train_step = tf.train.AdamOptimizer(learning_rate).minimize(xent, global_step=global_step)

	with tf.name_scope("accuracy"):
		error11 = tf.abs(logits-y)
		error22 = tf.abs(logits+6.28-y)
		error33 = tf.abs(logits-6.28-y)
		correct_prediction = tf.reduce_mean(tf.pow(tf.minimum(tf.minimum(error11, error22), error33), 2))
		# correct_prediction = tf.reduce_mean(tf.pow((logits-y), 2))
		accuracy = tf.cast(correct_prediction, tf.float32)
		accuracy_summary = tf.summary.scalar("accuracy", accuracy)

	summ = tf.summary.merge_all()

	saver = tf.train.Saver()
	save_dir = 'checkpoints/' + LOGDIR + hparam + '/'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	try:
		print('Trying:')
		last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
		saver.restore(sess, save_path=last_chk_path)
		print('success')
	except:
		print('failed')
		sess.run(tf.global_variables_initializer())

	train_writer = tf.summary.FileWriter(LOGDIR + hparam + "/train")
	train_writer.add_graph(sess.graph)
	test_writer = tf.summary.FileWriter(LOGDIR + hparam + "/test")

	for i in range(5000):
		batch = get_data("train", 1000, length)
		[i_global,train_accuracy, s] = sess.run([global_step,accuracy, summ], feed_dict={x:batch[0], y:batch[1]})
		train_writer.add_summary(s, i_global)
		if i_global % 5 == 0:
			test_batch = get_data("test", 100, length)
			# [test_accuracy] = sess.run([accuracy_summary], feed_dict={x:test_batch[0], y:test_batch[1]})
			[skull, layer1, layer2, print_accuracy, test_accuracy, y_estimation, y_truth] = sess.run([x, conv1, conv_out, accuracy, accuracy_summary, logits, y], feed_dict={x:test_batch[0], y:test_batch[1]})
			test_writer.add_summary(test_accuracy, i_global)
			print("%s \t %s" % (i_global, print_accuracy))
			saver.save(sess, save_path=save_dir, global_step=i_global)
			with open(LOGDIR + hparam + '/truthstep%s.csv' % i_global, "w") as output:		
				writer = csv.writer(output, lineterminator='\n')		
				for val in y_truth:		
					writer.writerow([val])
			with open(LOGDIR + hparam + '/estimationstep%s.csv' % i_global, "w") as output:		
				writer = csv.writer(output, lineterminator='\n')		
				for val in y_estimation:		
					writer.writerow([val])
			with open(LOGDIR + hparam + '/1conv%s.csv' % i_global, "w") as output:		
				writer = csv.writer(output, lineterminator='\n')		
				for val in layer1[0,:,0]:		
					writer.writerow([val]) 
			with open(LOGDIR + hparam + '/2conv%s.csv' % i_global, "w") as output:		
				writer = csv.writer(output, lineterminator='\n')		
				for val in layer2[0,:,0]:		
					writer.writerow([val]) 
			with open(LOGDIR + hparam + '/skull%s.csv' % i_global, "w") as output:		
				writer = csv.writer(output, lineterminator='\n')		
				for val in skull[0]:		
					writer.writerow([val]) 
		sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})	

def make_hparam_string(learning_rate, use_two_fc, use_two_conv):
	conv_param = "conv=2" if use_two_conv else "conv=3"
	fc_param = "fc=2" if use_two_fc else "fc=1"
	return "lr_%.0E,%s,%s" % (learning_rate, conv_param, fc_param)

def main():
	for learning_rate in [1E-2]:#[1E-3, 1E-4, 1E-5]:

		for use_two_fc in [False]:#[False, True]:
			for use_two_conv in [True]: #[True, False]:
				hparam = make_hparam_string(learning_rate, use_two_fc, use_two_conv)
				print('Starting run for %s' % hparam)

				cnn_model(learning_rate, use_two_fc, use_two_conv, hparam)
	print('Done training!')
	print('Run `tensorboard --logdir=%s` to see the results.' % LOGDIR)
	print('Running on mac? If you want to get rid of the dialogue asking to give '
				'network permissions to TensorBoard, you can provide this flag: '
				'--host=localhost')

if __name__ == '__main__':
	main()
	