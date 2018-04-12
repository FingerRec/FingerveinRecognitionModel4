import os
import tensorflow as tf
import numpy as np
import scipy.io
from numpy import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # close the warning

imagesize = 40*80
classNum = 50
trainNum = 250
testNum = 50
nnlambda = 0.4

train = 'imageTrainData.mat'
trainData = scipy.io.loadmat(train)['trainFeatures'].ravel()
trainData = np.reshape(trainData,[imagesize ,trainNum ])
trainData = np.transpose(trainData)

trainl = 'trainLabel.mat'
trainLabel = scipy.io.loadmat(trainl)['trainLabel'].ravel()
trainLabel = np.reshape(trainLabel,[trainNum, classNum ])

test = 'imageTestData.mat'
testData = scipy.io.loadmat(test)['testFeatures'].ravel()
testData = np.reshape(testData,[imagesize ,classNum ])
testData = np.transpose(testData)

testl = 'testLabel.mat'
testLabel = scipy.io.loadmat(testl)['testLabel'].ravel()
testLabel = np.reshape(testLabel,[50,classNum ])

# variable
x = tf.placeholder("float", [None, imagesize ])
W = tf.Variable(tf.zeros([imagesize , classNum ]))
b = tf.Variable(tf.zeros([classNum ]))

y = tf.nn.softmax(tf.sparse_matmul(x, W)+b)
y_ = tf.placeholder("float", [None, classNum ])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(nnlambda).minimize(cross_entropy)
# initialize variable
init = tf.global_variables_initializer()
# launch the module in a session
sess = tf.Session()
sess.run(init)
# img_batch, label_batch = tf.train.shuffle_batch([trainData, trainLabel]
for i in range(1000):
    sess.run(train_step, feed_dict={x: trainData , y_: trainLabel})

# evaluate model

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# take the mean
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print sess.run(accuracy, feed_dict={x: testData, y_: testLabel})