import tensorflow as tf


a = tf.placeholder(tf.float32)

sess = tf.Session()

# assign value to dummy node (tensor)
print(sess.run(a,{a: 3}))

# add 2 nodes
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

add = a+b

print(sess.run(add,{a: 3,b:5}))
