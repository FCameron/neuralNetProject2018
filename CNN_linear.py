from input_image_1D import get_data
import tensorflow as tf 
sess = tf.InteractiveSession()

length = 1000
nLabel = 63
nLayer1 = 64
nLayer2 = 128
nLayerFC = 1024

def conv_layer(input, size_in, size_out, name="conv"):
	with tf.name_scope(name):
		w = tf.Variable(tf.truncated_normal([5, size_in, size_out], stddev=0.1), name="W")
		b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
		conv = tf.nn.conv1d(input, w, stride=1, padding='SAME')
		act = tf.nn.relu(conv + b)
		tf.summary.histogram("weights", w)
		tf.summary.histogram("biases", b)
		tf.summary.histogram("activations", act)
		return tf.layers.max_pooling1d(act, pool_size=2, strides=2, padding='SAME')

def fc_layer(input, size_in, size_out, name="fc"):
	with tf.name_scope(name):
		w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
		b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
		act = tf.nn.relu(tf.matmul(input, w) + b)
		tf.summary.histogram("weights", w)
		tf.summary.histogram("biases", b)
		tf.summary.histogram("activations", act)
		return act

x = tf.placeholder(tf.float32, shape=[None, length], name="x")
x_image = tf.reshape(x, [-1, length, 1])
y = tf.placeholder(tf.float32, shape=[None, nLabel], name="labels")

conv1 = conv_layer(x_image, 1, nLayer1, "conv1")
conv2 = conv_layer(conv1, nLayer1, nLayer2, "conv2")

flattened = tf.reshape(conv2, [-1, 250*nLayer2])
fc1 = fc_layer(flattened, 250*nLayer2, nLayerFC, "fc1")
logits = fc_layer(fc1, nLayerFC, nLabel, "fc2")

with tf.name_scope("xent"):
	xent = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
tf.summary.scalar('cross_entropy', xent)

with tf.name_scope("train"):
	train_step = tf.train.AdamOptimizer(1e-4).minimize(xent)

with tf.name_scope("accuracy"):
	correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accruacy', accuracy)

sess.run(tf.global_variables_initializer())

merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("/tmp/neuralNetProject2018/5")
writer.add_graph(sess.graph)




for i in range(2000):
	batch = get_data('train', 100)

	if i % 5 == 0:
		s = sess.run(merged_summary, feed_dict={x:batch[0], y: batch[1]})
		writer.add_summary(s, i)
		print(i)
	sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})	

testset = get_data('test', 100)
print("test accuracy %g"%accuracy.eval(feed_dict={x: testset[0], y: testset[1]}))