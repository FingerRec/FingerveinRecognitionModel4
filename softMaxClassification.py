import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.sparse_matmul(x, W)+b)
y_ = tf.placeholder("float", [None, 10])

# launch the module in a session
#sess = tf.Session()
sess = tf.InteractiveSession()

# cross-entropy cost function
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# use backpropagation algorithm to minimize, 0.01: learning rate
# gradient descent algorithm
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# initialize variable
#init = tf.global_variables_initializer()
# evaluate model
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('log/train', sess.graph)
test_writer = tf.summary.FileWriter('log/test')
tf.global_variables_initializer().run()

#sess.run(init)
for i in range(1000):
    summary, acc = sess.run([merged, accuracy], feed_dict={x:mnist.test.images, y_:mnist.test.labels})
    test_writer.add_summary(summary, i)

    batch_xs, batch_ys = mnist.train.next_batch(100)
    summary, _ = sess.run([merged,train_step], feed_dict={x: batch_xs, y_: batch_ys})
    train_writer.add_summary(summary, i)

train_writer.close()
test_writer.close()
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
