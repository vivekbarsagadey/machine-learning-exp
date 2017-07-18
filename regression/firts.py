import tensorflow as tf
import matplotlib.pyplot as plt

X = [1,2,3]
Y = [1,2,3]

W = tf.placeholder(tf.float32)


# Our hypothesis for linear model
hypothesis = X * W


# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Variables for plotting cost function
W_val = []
cost_val = []

learning_rate = 0.1

for i in range(-10 , 50):
    feed_W = i*learning_rate
    curr_cost, cur_w = sess.run([cost, W],feed_dict={W:feed_W})
    W_val.append(cur_w)
    cost_val.append(curr_cost)


print('cost_val' , cost_val)

for i in cost_val:
   print(cost_val[i])

# Show the cost function
plt.plot(W_val, cost_val)
plt.show()
