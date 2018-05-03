# @cgiarrusso
from input_image_1D import get_data
import tensorflow as tf 
import os
import os.path

length = 1000
nLabel = 34
LOGDIR = "tensorboardphasetry1/"
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

	with tf.name_scope("xent"):
		xent = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(
				labels=y, logits=logits), name="xent")
		tf.summary.scalar("xent", xent)

	with tf.name_scope("train"):
		train_step = tf.train.AdamOptimizer(learning_rate).minimize(xent)

	with tf.name_scope("accuracy"):
		correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		accuracy_summary = tf.summary.scalar("accuracy", accuracy)

	summ = tf.summary.merge_all()

	sess.run(tf.global_variables_initializer())
	train_writer = tf.summary.FileWriter(LOGDIR + hparam + "/train")
	train_writer.add_graph(sess.graph)
	test_writer = tf.summary.FileWriter(LOGDIR + hparam + "/test")

	for i in range(2001):
		batch = get_data("train", 100, length)
		if i % 5 == 0:
			[train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x:batch[0], y:batch[1]})
			train_writer.add_summary(s, i)
		if i % 50 == 0:
			test_batch = get_data("test", 100, length)
			test_accuracy = sess.run(accuracy_summary, feed_dict={x:test_batch[0], y:test_batch[1]})
			test_writer.add_summary(test_accuracy, i)
			print(i)
		sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})	

def make_hparam_string(learning_rate, use_two_fc, use_two_conv):
	conv_param = "conv=2" if use_two_conv else "conv=3"
	fc_param = "fc=2" if use_two_fc else "fc=1"
	return "lr_%.0E,%s,%s" % (learning_rate, conv_param, fc_param)

def main():
	for learning_rate in [1E-4]:#[1E-3, 1E-4, 1E-5]:

		for use_two_fc in [True]:#[False, True]:
			for use_two_conv in [True]:#[True, False]:
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
	