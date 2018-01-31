import tensorflow as tf

# # create a constant op
# m1 = tf.constant([[3,3]])
# m2 = tf.constant([[2],[3]])
#
# # matrix m1 multi m2
# product = tf.matmul(m1,m2)
#
# print(product)
#
# # create a session, start the default gragh
# # sess = tf.Session()
# # result = sess.run(product)
# # print(result)
# # sess.close()
#
# with tf.Session() as sess:
#     result = sess.run(product)
#     print(result)

import numpy as np

x_data = np.random.rand(100)
y_data = x_data * 0.1 + 0.2

b = tf.Variable(0.)
k = tf.Variable(0.)
y = k * x_data + b

# define a loss function
loss = tf.reduce_mean(tf.square(y_data - y))
# define a GD
optimizer = tf.train.GradientDescentOptimizer(0.2)
# define a minimum loss function:
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step%20 == 0:
            print(step,sess.run([k,b]))