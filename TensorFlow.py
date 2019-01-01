import input_data
import numpy as np
import tensorflow as tf
import time

#setting parameters  
BATCH_SIZE = 64
LR = 1e-4
np.random.seed(120)

#loading database and converting to variables
mnist = input_data.read_data_sets('MNIST_data', one_hot=True) 
x = tf.placeholder(tf.float32, [None, 784])

y_ = tf.placeholder(tf.float32, [None, 10])

#building model
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# Conv layer 1 output shape (6,28,28)
W_conv1 = weight_variable([5, 5, 1, 6])
b_conv1 = bias_variable([6])
x_image = tf.reshape(x,[-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)

# Pooling layer 1 (max pooling) output shape (6,14,14)
h_pool1 = max_pool_2x2(h_conv1)

# Conv layer 2 output shape (16,14,14)
W_conv2 = weight_variable([5, 5, 6, 16])
b_conv2 = bias_variable([16])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# Pooling layer 2 (max pooling) output shape (16,7,7)
h_pool2 = max_pool_2x2(h_conv2)

# Fully connected layer 1 input shape (16*7*7)=(784)
W_fc1 = weight_variable([7 * 7 * 16, 120])
b_fc1 = bias_variable([120])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*16])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Fully connected layer 2 to shape (120) for 10 classes
W_fc2 = weight_variable([120, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#Compile model
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(LR).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#train && test
def train(EPOCH):
    count = 1
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('Test accuracy before train: %g%%' % (100*accuracy.eval(feed_dict={
                x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))
        for i in range(938*EPOCH):
            if(i%938==0):
                  print("--------------epoch "+str(count)+"--------------")
                  time_start=time.time()
                  count += 1
            batch = mnist.train.next_batch(BATCH_SIZE)
            if i % 100 == 99:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('batch: %d training accuracy: %g%%' % (i+1, 100*train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            if(i%937==0 and i!=0):
                time_end=time.time()
                print('time cost: ',time_end-time_start,'s')
        print('Test accuracy after train Accuracy: %g%%' % (100*accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))


        






    
