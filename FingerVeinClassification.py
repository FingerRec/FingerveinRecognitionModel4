import os
import tensorflow as tf
import numpy as np
import scipy.io
from numpy import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # close the warning

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

# variable
x = tf.placeholder("float", [None, featurelNum ])
W = tf.Variable(tf.zeros([featurelNum , classNum ]))
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
for i in range(4000):
#    batch_xs, batch_ys = trainData[1:11, :], trainLabel[1:11]
    sess.run(train_step, feed_dict={x: trainData , y_: trainLabel})
#    trainData, trainLabel = sess.run(train_step,trainData,trainLabel)

# evaluate model

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# take the mean
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print sess.run(accuracy, feed_dict={x: testData, y_: testLabel})