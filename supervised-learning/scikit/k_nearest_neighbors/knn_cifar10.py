import _pickle as pickle
import scipy.spatial.distance as sp
import numpy as np
import os
import time
import math


class DocumentLoader:

    def __init__(self, name = "test"):
        self.name = name

    # Load single batch of cifar
    def load_cifar_batch(self,filename):
        with open(filename, 'rb') as f:
            datadict = pickle.load(f, encoding='latin1')
        return datadict['data'].astype(np.float64), np.array(datadict['labels'])

    # Load all of cifar
    def load_cifar(self,folder):
        with open(os.path.join(folder, 'batches.meta'), 'rb') as f:
            names = pickle.load(f, encoding='latin1')
        training_data = np.empty([50000, 3072], dtype=np.float64)
        training_labels = np.empty([50000], dtype=np.uint8)
        for i in range(1, 6):
            start = (i - 1) * 10000
            end = i * 10000
            training_data[start:end], training_labels[start:end] = \
                self.load_cifar_batch(os.path.join(folder, 'data_batch_%d' % i))
        testing_data, testing_labels = self.load_cifar_batch(os.path.join(folder, 'test_batch'))
        training_data_grayscale = training_data.reshape((50000, 3, 1024)).transpose((0, 2, 1))
        training_data_grayscale = np.mean(training_data_grayscale, axis=2)
        testing_data_grayscale = testing_data.reshape((10000, 3, 1024)).transpose((0, 2, 1))
        testing_data_grayscale = np.mean(testing_data_grayscale, axis=2)
        return training_data, training_data_grayscale, training_labels, testing_data, testing_data_grayscale, testing_labels, names['label_names']

    # Load part of cifar for cross validation
    def load_cifar_cross_validation(self,folder, i):
        td = np.empty([40000, 3072], dtype=np.float64)
        tl = np.empty([40000], dtype=np.uint8)
        for j in range(1, 6):
            if i != j:
                if j < i:
                    diff = 1
                else:
                    diff = 2
                start = (j - diff) * 10000
                end = (j - diff + 1) * 10000
                td[start:end, :], tl[start:end] = \
                    self.load_cifar_batch(os.path.join(folder, 'data_batch_%d' % j))
        vd, vl = self.load_cifar_batch(os.path.join(folder, 'data_batch_%d' % i))
        return td, tl, vd, vl


class KNN(object):

    def __init__(self):
        pass

    def train(self, data, labels):
        # data is N x D where each row is a data point. labels is 1-dimension of size N
        # KNN classifier simply remembers all the training data
        self.training_data = data
        self.training_labels = labels

    def predict(self, data, k):
        # data is M x D where each row is a data point, k is the number of nearest neighbours
        # y_predict is the predicted labels of data
        y_predict = np.zeros(data.shape[0], dtype=self.training_labels.dtype)
        self.process(data, k, y_predict)
        return y_predict



    def process(self, data, k, y_pred):
        # data is M x D where each row is a data point, k is the number of nearest neighbours, y_pred is the predicted labels of data
        # (a + b)^2 = a^2 + b^2 - 2ab
        a_sum_square = np.sum(np.square(self.training_data), axis=1)
        b_sum_square = np.sum(np.square(data), axis=1)
        two_a_dot_bt = 2 * np.dot(self.training_data, data.T)
        # Compute Euclidean distance, distances is N x M where each column 'i' is the distances of the ith data point from the training data points
        distances = np.sqrt(a_sum_square[:, np.newaxis] + b_sum_square - two_a_dot_bt)
        for i in range(data.shape[0]):
            # Get ith column of distances and continue operations on it as normal (get lowest k)
            curr_distance = distances[:, i]
            # Get the k indexes corresponding to the lowest distances
            min_idx = np.argpartition(curr_distance, k)[0:k]
            # Get the votes
            votes = self.training_labels[min_idx]
            # Count the votes
            labels_count = np.bincount(votes)
            # Choose the majority vote
            y_pred[i] = np.argmax(labels_count)




def train_and_predict(xtr, ytr, xte, yte, k, color, names):
    knn_o = KNN()
    knn_o.train(xtr, ytr)
    predict = knn_o.predict(xte, k)
    return predict

"""
Load all the data
"""




xtr, xgtr, ytr, xte, xgte, yte, names = DocumentLoader().load_cifar('../../../data/CIFAR-10/cifar-10-batches-py')

print("names" , names)

k = 1
start = time.time()
predict = train_and_predict(xtr, ytr, xte, yte, k, 'colored', names)

print("predict \n ",predict)


