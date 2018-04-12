# mnist cnn 99.2%
import tensorflow as tf
import os
import numpy as np
import scipy.io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning

# --------------------load data-----------------------------------------------------------
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

featurelNum = 400
classNum = 50
trainNum = 250
testNum = 50
nnlambda = 0.009

train = 'trainData.mat'
trainData = scipy.io.loadmat(train)['trainFeatures'].ravel()
trainData = np.reshape(trainData,[featurelNum ,trainNum ])
trainData = np.transpose(trainData)

trainl = 'trainLabel.mat'
trainLabel = scipy.io.loadmat(trainl)['trainLabel'].ravel()
trainLabel = np.reshape(trainLabel,[trainNum, classNum ])

test = 'testData.mat'
testData = scipy.io.loadmat(test)['testFeatures'].ravel()
testData = np.reshape(testData,[featurelNum ,classNum ])
testData = np.transpose(testData)

testl = 'testLabel.mat'
testLabel = scipy.io.loadmat(testl)['testLabel'].ravel()
testLabel = np.reshape(testLabel,[50,classNum ])

# ----------------construct session and initialize-----------------------------------------
sess = tf.InteractiveSession()
x = tf.placeholder("float", shape=[None, 400])
y_ = tf.placeholder("float", shape=[None, 50])
W = tf.Variable(tf.zeros([400, 50]))
b = tf.Variable(tf.zeros([50]))
#summary_op = tf.merge_all_summaries()
# sess.run(tf.initialize_all_variables())
# ---weight initialization,use 0.1 to prevent 0 gradients-------------------------
# 0.1 to prevent 0 gradient

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# ---Convolution and Pooling --------------------------------------------------
#   1stride size,0 padding size
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


#  block:2x2
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')

# ---First Convolution Layer-------------------------------------
# patch:5x5  32features 1:input channel 32:output channels
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
# x to 4d,28x28 image width and height,1:color channels
x_image = tf.reshape(x, [-1, 20, 20, 1])
# ReLu function
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# ----Second Convolutional Layer--------------------------------------
# 64:output channels
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# ----Densely Connected Layer----------------------------------------
# image size:7x7  neurons number:1024 ReLu function
W_fc1 = weight_variable([5*5*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 5*5*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# --Dropout,reduce overfitting
# a place-holder for thr probability kept during dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# ---Readout Layer
W_fc2 = weight_variable([1024, 50])
b_fc2 = bias_variable([50])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# --Train and Evaluate the Model
with tf.name_scope('cross_entropy'):
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
tf.summary.scalar('cross_entropy', cross_entropy)

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('log/train', sess.graph)
test_writer = tf.summary.FileWriter('log/test')

tf.global_variables_initializer().run()

# summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def = sess.graph_def)
for i in range(4000):
 #   batch = mnist.train.next_batch(50)
    if i % 100 == 0:
 #       summary_str = sess.run(summary_op,feed_dict=feed_dict)
 #       summary_writer.add_sumary(summary_str,step);
        train_accuracy = accuracy.eval(feed_dict={
            x: trainData , y_: trainLabel, keep_prob: 1.0})
        print "setup_%d,_training_accuracy%g" % (i, train_accuracy)
        print "test_accuracy_%g" % accuracy.eval(feed_dict={
            x: testData, y_: testLabel, keep_prob: 1.0})
    summary, _ = sess.run([merged, train_step], feed_dict={x: trainData, y_: trainLabel, keep_prob: 0.5})
    train_writer.add_summary(summary, i)
    #train_step.run(feed_dict={x: trainData, y_: trainLabel, keep_prob: 0.5})
