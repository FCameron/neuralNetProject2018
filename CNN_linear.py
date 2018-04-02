from input_image_1D import get_data
import tensorflow as tf 
sess = tf.InteractiveSession()

length = 1000
nLabel = 63

### Variables
x = tf.placeholder(tf.float32, shape=[None, length])
y = tf.placeholder(tf.float32, shape=[None, nLabel])

### Convolution and Pooling
def conv1d(x, W):
	return tf.nn.conv1d(x, W, stride=2, padding='SAME')

def max_pool(x):
	return tf.layers.max_pooling1d(x, pool_size=2, strides=2, padding='SAME')

### Weights and Biases Initialization
def weight_helper(shape):
	initial = tf.random_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_helper(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

### Convolutional Layer 1
W_conv1 = weight_helper([5, 1, 64])
b_conv1 = bias_helper([64])
x_image = tf.reshape(x, [-1, length, 1])

h_conv1 = tf.nn.relu(conv1d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool(h_conv1)

### Convolutional Layer 2
W_conv2 = weight_helper([5, 64, 128])
b_conv2 = bias_helper([128])

h_conv2 = tf.nn.relu(conv1d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool(h_conv2)

### Fully-connected Layer
W_fc1 = weight_helper([63*128, 1024])
b_fc1 = bias_helper([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 63*128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

### Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

### Readout Layer
W_fc2 = weight_helper([1024, nLabel])
b_fc2 = bias_helper([nLabel])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

### Train and Evaluate the Model
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy) # 1e-4
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

### Saver Function
# saver = tf.train.Saver()
# save_dir = 'checkpoints/'
# if not os.path.exists(save_dir):
# 	or.makedirs(save_dir)
# save_path = os.path.join(save_dir, 'best_validation')

sess.run(tf.global_variables_initializer())

for i in range(1000):
	batch = get_data('train', 20)
	if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
		print("step %d, training accuracy %g"%(i, train_accuracy))
		# saver.save(sess=session, save_path=save_path)
	train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})

testset = get_data('test', 30)
print("test accuracy %g"%accuracy.eval(feed_dict={x: testset[0], y: testset[1], keep_prob: 1.0}))



# saver.restore(sess=session, save_path=save_path)
