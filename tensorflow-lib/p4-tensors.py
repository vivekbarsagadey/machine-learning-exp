import pandas as pd
import tensorflow as tf
import numpy as np
import pprint

tf.set_random_seed(777)

pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

#Simple Array

t = np.array([0., 1., 2., 3., 4., 5., 6.])
pp.pprint(t)
print(t.ndim) # rank
print(t.shape) # shape
print(t[0], t[1], t[-1])
print(t[2:5], t[4:-1])
print(t[:2], t[3:])

# 2D Array

t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
pp.pprint(t)
print("rank >> ",t.ndim) # rank
print("shape >> ",t.shape) # shape

#Shape, Rank, Axis
t = tf.constant([1,2,3,4])
print(t , " >>  shape >>>" ,tf.shape(t).eval())

t = tf.constant([[1,2],
                 [3,4]])
print(tf.shape(t).eval())

t = tf.constant(
    [
        [
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]
            ],
            [
                [13, 14, 15, 16],
                [17, 18, 19, 20],
                [21, 22, 23, 24]
            ]
        ]
    ])
print(tf.shape(t).eval())

"""
reduce_sum for axis  
"""
t1 = tf.constant([[[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8]],

       [[10, 11, 12],
        [13, 14, 15],
        [16, 17, 18]]])


print(t1.shape)
print("reduce_sum for axis 0 ", tf.reduce_sum(t1, axis=0).eval())
print("reduce_sum for axis 1 ",tf.reduce_sum(t1, axis=1).eval())
print("reduce_sum for axis 2 ",tf.reduce_sum(t1, axis=2).eval())
print("reduce_sum for axis all ", tf.reduce_sum(t1).eval())


"""
Matmul VS multiply
"""
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
print(" matmul of matrix1 and matrix2 >> ",tf.matmul(matrix1, matrix2).eval())

print(" * of matrix1 and matrix2 >> ",(matrix1*matrix2).eval())


matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
print(" + of matrix1 and matrix2 >> ",(matrix1+matrix2).eval())

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2., 2.]])
print(" + of matrix1 and matrix2 >> ",(matrix1+matrix2).eval())


"""
Random values for variable initializations
"""
print("random_normal([3])" , tf.random_normal([3]).eval())


print("random_uniform([2])" ,tf.random_uniform([2]).eval())

print("random_uniform([2, 3])" ,tf.random_uniform([2, 3]).eval())


"""
Argmax with axis
"""

x = [[0, 1, 2],
     [2, 1, 0]]
print("argmax with axis 0 " ,tf.argmax(x, axis=0).eval())
print("argmax with axis 1 " ,tf.argmax(x, axis=1).eval())

"""
Reshape, squeeze, expand_dims
"""
t = np.array([[[0, 1, 2],
               [3, 4, 5]],
              [[6, 7, 8],
               [9, 10, 11]]])
print("the shape is " ,t.shape)
print("tf.reshape(t, shape=[-1, 3]) :: \n", tf.reshape(t, shape=[-1, 3]).eval())
print("tf.reshape(t, shape=[-1, 1, 3]) :: \n", tf.reshape(t, shape=[-1, 1, 3]).eval())

print("tf.squeeze([[0], [1], [2]]) :: \n", tf.squeeze([[0], [1], [2]]).eval())


print("tf.expand_dims([0, 1, 2], 1) :: \n", tf.expand_dims([0, 1, 2], 1).eval())

"""
One hot
"""
print("tf.one_hot([[0], [1], [2], [0]], depth=3) :: \n", tf.one_hot([[0], [1], [2], [0]], depth=3).eval())
t = tf.one_hot([[0], [1], [2], [0]], depth=3)
print("tf.one_hot([[0], [1], [2], [0]], depth=3) with reshape :: \n",tf.reshape(t, shape=[-1, 3]).eval())

"""
casting
"""
print("cast >>> \n", tf.cast([1.8, 2.2, 3.3, 4.9,1.8], tf.int32).eval())
print("cast >>> \n", tf.cast([True, False, 1 == 1, 0 == 1], tf.int32).eval())

"""
Stack
"""

x = [1, 4]
y = [2, 5]
z = [3, 6]

# Pack along first dim.
print("stack >>> \n",tf.stack([x, y, z]).eval())
print("tf.stack([x, y, z], axis=1).eval()", tf.stack([x, y, z], axis=1).eval())

"""
Ones like and Zeros like
"""
x = [[0, 1, 2],
     [2, 1, 0]]

print("ones_like >>> \n",tf.ones_like(x).eval())
print("zeros_like >>> \n",tf.zeros_like(x).eval())

"""
Zip
"""
for x, y in zip([1, 2, 3], [4, 5, 6]):
    print(x, y)

for x, y, z in zip([1, 2, 3], [4, 5, 6], [7, 8, 9]):
    print(x, y, z)


"""
Transpose
"""
t = np.array([
    [
        [0, 1, 2],
        [3, 4, 5]
    ],
    [
        [6, 7, 8],
        [9, 10, 11]
    ]
])
pp.pprint(t.shape)
pp.pprint(t)

t1 = tf.transpose(t, [1, 0, 2])
pp.pprint(sess.run(t1).shape)
pp.pprint(sess.run(t1))