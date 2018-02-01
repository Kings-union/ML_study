# coding: utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)


# define the size of batch
batch_size = 100
n_batch = mnist.train.num_examples // batch_size

# create a namespace
with tf.name_scope('input'):
    # define the input and output layer
    x = tf.placeholder(tf.float32,[None,784],name='x_input') # 784 = 28 * 28
    y = tf.placeholder(tf.float32,[None,10],name='y_input')


with tf.name_scope('layer'):
    with tf.name_scope('weights'):
        # define the NN layer
        W = tf.Variable(tf.zeros([784,10]),name='W')
    with tf.name_scope('bias'):
        b = tf.Variable(tf.zeros([10]),name='b')
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x,W) + b
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx_plus_b)

loss = tf.reduce_mean(tf.square(y-prediction))
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1)) # argmax return a max number of the matrix

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/',sess.graph)
    for epoch in range(51):
        for batch in range(n_batch):
            batch_xs,batch_ys =  mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})

        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))

