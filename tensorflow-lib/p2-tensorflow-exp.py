#Importing TensorFlow

#This gives Python access to all of TensorFlow's classes, methods, and symbols.
import tensorflow as tf


#The Computational Graph
'''
You might think of TensorFlow Core programs as consisting of two discrete sections:

    Building the computational graph.
    Running the computational graph.

'''

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)


# creates a Session
sess = tf.Session()
print(sess.run([node1, node2]))

addNode = tf.add(node1,node2)
print(sess.run(addNode))



sess.close()