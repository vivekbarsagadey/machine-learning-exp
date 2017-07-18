import tensorflow as tf

W = tf.Variable([.3], dtype=tf.float32 , name='W')
b = tf.Variable([-.3], dtype=tf.float32, name='b')
x = tf.placeholder(tf.float32 , name='x')
#current model
linear_model = W * x + b;



sess = tf.Session()


init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(linear_model, {x:[1,2,3,4]}))

#To evaluate the model on training data, provided data
y = tf.placeholder(tf.float32 , name='y')


#standard loss model for linear regression :  sums the squares of the deltas between the current model and the provided data

squared_deltas = tf.square(linear_model - y , name='squared_deltas')

# sum of squared_deltas = loss
loss = tf.reduce_sum(squared_deltas , name='loss')

# apply value
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

# apply some other for reduce loss
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))


#writer = tf.train.SummaryWriter("c://logs/abc/test", graph=tf.get_default_graph())
writer = tf.summary.FileWriter("c://logs/abc/test", graph=tf.get_default_graph())



