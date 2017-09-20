#
#   iris_lr_tensorflow.py
#       date. 5/6/2016
#       IRIS data set classification
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import tensorflow as tf

def prep_data(target_class=1, train_siz=120, test_siz=30):
    '''
      class: 
        1. Iris Setosa, 2. Iris Versicolor, 3. Iris Virginica
    '''
    cols = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']  
    iris_df = pd.read_csv('../../data/iris.data', header=None, names=cols)
    
    # Encode class 
    class_name = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    iris_df['iclass'] = [class_name.index(class_str) 
                            for class_str in iris_df['class'].values]
    
    # Random Shuffle before split to train/test
    orig = np.arange(len(iris_df))
    perm = np.copy(orig)
    np.random.shuffle(perm)
    iris = iris_df[['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'iclass']].values
    iris[orig, :] = iris[perm, :]
    
    # Arrange Label value to consider one vs. all classification
    # ex. class 0 --> label 1.0, class 1 or 2 --> label 0.0
    if target_class in [1, 2, 3]:
        target_class = target_class - 1  # python indexing rule
        for i in range(len(iris)):
            iclass = int(iris[i, -1])
            iris[i, -1] = float(iclass == target_class)
    else:
        print('Value Error (target_class)')

    # Split dataset
    trX = iris[:train_siz, :-1]
    teX = iris[train_siz:, :-1]
    trY = iris[:train_siz, -1]
    teY = iris[train_siz:, -1]
    
    return trX, trY, teX, teY
    
def linear_model(X, w, b):
    output = tf.matmul(X, w) + b

    return output

if __name__ == '__main__':
    tr_x, tr_y, te_x, te_y = prep_data(target_class=1)

    # Variables
    x = tf.placeholder(tf.float32, [None, 4])
    y_ = tf.placeholder(tf.float32, [None, 1])
    
    p5 = tf.constant(0.5)  # threshold of Logistic Regression

    w = tf.Variable(tf.random_normal([4, 1], mean=0.0, stddev=0.05))
    b = tf.Variable([0.])

    y_pred = linear_model(x, w, b)
    y_pred_sigmoid = tf.sigmoid(y_pred)   # for prediction

    # cross_entropy = -tf.reduce_sum(y_ * tf.log(y_pred_sigmoid))
    x_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_)
    loss = tf.reduce_mean(x_entropy)
    
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    delta = tf.abs((y_ - y_pred_sigmoid))
    correct_prediction = tf.cast(tf.less(delta, p5), tf.int32)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Train
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        print('Training...')
        for i in range(5001):
            batch_xs, batch_ys = tr_x, tr_y
            fd_train = {x: batch_xs, y_: batch_ys.reshape((-1, 1))}
            train_step.run(fd_train)                 
        
            if i % 500 == 0:
                loss_step = loss.eval(fd_train)
                train_accuracy = accuracy.eval(fd_train)
                print('  step, loss, accurary = %6d: %8.3f,%8.3f' % (i, 
                                                loss_step, train_accuracy))      
        # Test trained model
        fd_test = {x: te_x, y_: te_y.reshape((-1, 1))}
        print('accuracy = %10.4f' % accuracy.eval(fd_test))
        writer = tf.summary.FileWriter("c://logs/abc/logic", graph=tf.get_default_graph())

