import tensorflow as tf

# x = tf.Variable([1,2])
# a = tf.constant([3,3])
#
# init = tf.global_variables_initializer()
#
# # Add a sub op
# sub = tf.subtract(x,a)
#
# # Add a add op
# add = tf.add(x,sub)
#
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(sub))
#     print(sess.run(add))

# create a variable initial it to 0
state = tf.Variable(0,name='counter')
new_value = tf.add(state,1)
update = tf.assign(state,new_value)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))