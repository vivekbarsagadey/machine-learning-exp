import tensorflow as tf

W = tf.Variable([.3], dtype=tf.float32 , name='W')
b = tf.Variable([-.3], dtype=tf.float32, name='b')
x = tf.placeholder(tf.float32 , name='x')

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

#To evaluate the model on training data, provided data
y = tf.placeholder(tf.float32 , name='y')
#current model
H = W * x + b;

# sum of squared_deltas = loss
cost = tf.reduce_sum(tf.square(H - y , name='squared_deltas') , name='cost')

# apply value
print(sess.run(cost, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

writer = tf.summary.FileWriter("../logs", graph=tf.get_default_graph())



