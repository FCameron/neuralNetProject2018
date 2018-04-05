from input_image_1D import get_data
import tensorflow as tf 
import os
import os.path

length = 1000
nLabel = 13
LOGDIR = "tensorboard/"
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
		embedding_input = relu
		tf.summary.histogram("fc1/relu", relu)
		embedding_size = 1024
		logits = fc_layer(fc1, 1024, nLabel, "fc2")
	else:
		embedding_input = flattened
		embedding_size = 250*64
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
		tf.summary.scalar("accuracy", accuracy)

	with tf.name_scope("test_accuracy"):


	summ = tf.summary.merge_all()

	embedding = tf.Variable(tf.zeros([308, embedding_size]), name="test_embedding")
	assignment = embedding.assign(embedding_input)
	saver = tf.train.Saver()

	sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter(LOGDIR + hparam)
	writer.add_graph(sess.graph)
	testset = get_data("test", 308, tsvlocation=LOGDIR+hparam+'/'+LABELS)
	config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
	embedding_config = config.embeddings.add()
	embedding_config.tensor_name = embedding.name
	embedding_config.metadata_path = LABELS
	tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

	for i in range(2000):
		batch = get_data("train", 100)
		if i % 5 == 0:
			[train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x:batch[0], y: batch[1]})
			writer.add_summary(s, i)
		if i % 500 == 0:
			print(i)
			sess.run(assignment, feed_dict={x: testset[0], y: testset[1]})
			saver.save(sess, os.path.join(LOGDIR+hparam+"/", "model.ckpt"), i)
		sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})	

def make_hparam_string(learning_rate, use_two_fc, use_two_conv):
	conv_param = "conv=2" if use_two_conv else "conv=3"
	fc_param = "fc=2" if use_two_fc else "fc=1"
	return "lr_%.0E,%s,%s" % (learning_rate, conv_param, fc_param)

def main():
	for learning_rate in [1E-3, 1E-4, 1E-5]:

		for use_two_fc in [False, True]:
			for use_two_conv in [True, False]:
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
	