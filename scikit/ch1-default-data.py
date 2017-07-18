from sklearn import datasets

import numpy as np
import matplotlib.pyplot as plt
import logging
from pprint import pprint


class DataSetTest():

    """
    We can find out all the data sets
    C:\ProgramData\Anaconda3\Lib\site-packages\sklearn\datasets

    ===============================
    Toy datasets
    load_boston([return_X_y]) 	Load and return the boston house-prices dataset (regression).
    load_iris([return_X_y]) 	Load and return the iris dataset (classification).
    load_diabetes([return_X_y]) 	Load and return the diabetes dataset (regression).
    load_digits([n_class, return_X_y]) 	Load and return the digits dataset (classification).
    load_linnerud([return_X_y]) 	Load and return the linnerud dataset (multivariate regression).
    ===============================

    """

    def toy_datasets(self):
        logging.debug('--------------- toy datasets ---------')
        iris = datasets.load_iris()
        digits = datasets.load_digits()
        diabetes = datasets.load_diabetes()
        print('iris data   >', iris.data)
        print('digits data > ',digits.data)


        print('digits.target' , digits.target)

        print('digits.image' , digits.images[0])

        logging.debug('--------------- diabetes ---------')

        print('diabetes target ', diabetes.target)
        print('diabetes data ', diabetes.data)



    def sample_images_datasets(self):
        """
            Sample images
            load_sample_images() 	Load sample images for image manipulation.
            load_sample_image(image_name) 	Load the numpy array of a single sample image
        """

        logging.debug('--------------- Sample images ---------')

        images = datasets.load_sample_images()

        print(images)

        print("filenames >>>>>>>>>> \n",images.filenames)

        chinaImage = datasets.load_sample_image("china.jpg")
        print(chinaImage)

        '''
        
        
        Warning
        
        The default coding of images is based on the uint8 dtype to spare memory. Often machine learning algorithms work best if the input is converted to a floating point representation first. Also, if you plan to use matplotlib.pyplpt.imshow don’t forget to scale to the range 0 - 1 as done in the following example. 
        '''

        # Convert to floats instead of the default 8 bits integer coding. Dividing by
        # 255 is important so that plt.imshow behaves works well on float data (need to
        # be in the range [0-1])
        china = np.array(chinaImage, dtype=np.float64) / 255






    def sample_generators_datasets(self) :
        """
            Sample generators
        """

        logging.debug('----------------- Sample generators (Single label) -----------')

        blobs = datasets.make_blobs()
        print('blobs  for ' , blobs)

        print('classification' , datasets.make_classification())
        print('gaussian quantiles' , datasets.make_gaussian_quantiles())

        print('----------------- Sample generators ( Multilabel) -----------')

        print('multilabel_classification' , datasets.make_multilabel_classification())
        print('make_biclusters' , datasets.make_biclusters(shape=(300, 300) , n_clusters=5))
        print('make_checkerboard' , datasets.make_multilabel_classification())






    def generators_for_regression_datasets(self):
        """
            Generators for regression
            sparse random linear combination of random features, with noise
            make_sparse_uncorrelated
        """

        logging.debug('----------------- Generators for regression  -----------')
        print('sparse_uncorrelated ' , datasets.make_sparse_uncorrelated())





    def support_vector_machines_datasets(self):
        """
        Support Vector Machines (SVMs)

        <label> <feature-id>:<feature-value> <feature-id>:<feature-value>
        1 qid:2 1:0 2:0 3:1 4:0.2 5:0
        2 qid:2 1:1 2:0 3:1 4:0.4 5:0

        svmlight SVM Light is a C program by Thorsten Joachims that implements a support vector machine. provides several kernels, such as linear, polynomial, radial basis function, and sigmoid
        LIBSVM -- A Library for Support Vector Machines, It supports multi-class classification.
        """

        logging.debug('----------------- Support Vector Machines  -----------')
        X_train, y_train = datasets.load_svmlight_file("../data/svmlight/example3/train.dat")
        print("Support Vector Machines \n" , X_train, y_train)

        X_train, y_train, X_test, y_test = datasets.load_svmlight_files(("../data/svmlight/example3/train.dat","../data/svmlight/example3/test.dat"))
        print(' X_train ', X_train, 'y_train ',  y_train, ' X_test ', X_test, 'y_test ', y_test)


    """
    Loading from external datasets
    
        pandas.io provides tools to read data from common formats including CSV, Excel, JSON and SQL. DataFrames may also be constructed from lists of tuples or dicts. Pandas handles heterogeneous data smoothly and provides tools for manipulation and conversion into a numeric array suitable for scikit-learn.
        scipy.io specializes in binary formats often used in scientific computing context such as .mat and .arff
        numpy/routines.io for standard loading of columnar data into numpy arrays
        scikit-learn’s datasets.load_svmlight_file for the svmlight or libSVM sparse format
        scikit-learn’s datasets.load_files for directories of text files where the name of each directory is the name of each category and each file inside of each directory corresponds to one sample from that category
    
    For some miscellaneous data such as images, videos, and audio, you may wish to refer to:
    
        skimage.io or Imageio for loading images and videos to numpy arrays
        scipy.misc.imread (requires the Pillow package) to load pixel intensities data from various image file formats
        scipy.io.wavfile.read for reading WAV files into a numpy array
    
    """


    def news_groups_datasets(self):
        """

        20 newsgroups text dataset
        comprises around 18000 newsgroups posts on 20 topics split in two subsets: one for training and the other one for testing.

        """
        newsgroups_train = datasets.fetch_20newsgroups(subset='train')
        logging.debug('newsgroups_train' ,newsgroups_train)

        pprint(list(newsgroups_train.target_names))
        #pprint(list(newsgroups_train.data))


    def mldata_org_repository_datasets(self):
        """
        datasets from the mldata.org repository

        data from http://mldata.org/repository/data/

        """
        logging.debug('---------- mldata.org repository -------------')
        custom_data_home_path = "../data/svmlight/mldata.org/"
        mnist = datasets.fetch_mldata('MNIST original', data_home=custom_data_home_path)
        print('mnist.shape ' , mnist.data.shape)




dataSetTest = DataSetTest()
dataSetTest.toy_datasets()
dataSetTest.sample_images_datasets()
dataSetTest.sample_generators_datasets()
dataSetTest.generators_for_regression_datasets()
dataSetTest.support_vector_machines_datasets()
dataSetTest.news_groups_datasets()
dataSetTest.mldata_org_repository_datasets()