"""

Singular-value decomposition

This code will cover following thing:


    1> Calculate Singular-Value Decomposition
    2> Reconstruct Matrix from SVD
    3> SVD for Pseudoinverse
    4> SVD for Dimensionality Reduction


"""


"""

The Singular-Value Decomposition (SVD), is a matrix decomposition method for reducing a matrix to its constituent parts 
in order to make certain subsequent matrix calculations simpler.

The singular value decomposition (SVD) provides another way to factorize a matrix, into singular vectors and singular values.

A = U . Sigma . V^T

A is the real m x n matrix that we wish to decompose, U is an m x m matrix, 
Sigma is an m x n diagonal matrix (singular values), 
and V^T is the  transpose of an n x n matrix where T is a superscript.

"""


# Singular-value decomposition
from numpy import array
from scipy.linalg import svd


"""
Reconstruct Matrix from SVD
"""
def reconstructMatrix(A , s ,U , V):
    from numpy import dot
    from numpy import zeros
    from numpy import array
    from numpy import diag
    print( "Org Matrix is \n", A)
    print("A shape is ", A.shape)
    # create m x n Sigma matrix
    Sigma = zeros((A.shape[0], A.shape[1]))

    print("Sigma Matrix is \n", Sigma)
    print("Sigma shape is ",Sigma.shape)
    # populate Sigma with n x n diagonal matrix
    Sigma[:A.shape[1], :A.shape[1]] = diag(s)
    print("New Sigma Matrix is \n", Sigma)
    # reconstruct matrix  B = U . Sigma . V^T == A
    B = U.dot(Sigma.dot(V))
    return  B

"""
Pseudoinverse : Generalization of the matrix inverse for square matrices to rectangular matrices where the number of rows and columns are not equal
    A^+ = V . D^+ . U^T

where 
A^+ is the pseudoinverse
D^+ is the pseudoinverse of the diagonal matrix Sigma
U^T is the transpose of U.

pinv() for calculating the pseudoinverse of a rectangular matrix

"""
def pseudoInverse(A , s ,U , V):
    print(" ************************** Pseudoinverse : Generalization of the matrix "
          "inverse for square matrices to rectangular matrices "
          "where the number of rows and columns are not equal **********************")
    ''' Default method in numpy'''
    from numpy.linalg import pinv
    B = pinv(A)
    print("pseudoInverse B is \n", B)
    ''' Create over own method
     A^+ = V^T . D^T . U^V
    '''
    from numpy import zeros
    from numpy import diag
    # reciprocals of s
    d = 1.0 / s
    # create m x n D matrix
    D = zeros(A.shape)
    # populate D with n x n diagonal matrix
    D[:A.shape[1], :A.shape[1]] = diag(d)
    # calculate pseudoinverse
    C = V.T.dot(D.T).dot(U.T)
    print("pseudoInverse C is \n", C)
    print("A shape is ", A.shape)
    print("B shape is ", B.shape)
    print("C shape is ", C.shape)
    return C


"""
SVD for Dimensionality Reduction

Data with a large number of features, such as more features (columns) than observations (rows) may be reduced to a 
smaller subset of features that are most relevant to the prediction problem.

The result is a matrix with a lower rank that is said to approximate the original matrix.

To do this we can perform an SVD operation on the original data and select the top k largest singular values in Sigma. 
These columns can be selected from Sigma and the rows selected from V^T.

B = U . Sigmak . V^T k

Steps:
T = U . Sigmak
T = V^Tk . A

Note both T should be equal.

"""


def svd_dimensionality_reduction(A, s, U, V):
    print("***************** svd dimensionality reduction A Matrix with svd *********************")
    from numpy import diag
    from numpy import zeros
    # create m x n Sigma matrix
    Sigma = zeros((A.shape[0], A.shape[1]))
    # populate Sigma with n x n diagonal matrix
    Sigma[:A.shape[1], :A.shape[1]] = diag(s)
    # select
    n_elements = 2
    Sigma = Sigma[:, :n_elements]
    V = V[:n_elements, :]
    print("Initial V \n", V)
    # reconstruct
    B = U.dot(Sigma.dot(V))
    print("B (for checking B and A should be same) \n", B)

    # transform
    T = U.dot(Sigma)
    print("Transform T from U and Sigma k \n",T)
    T = A.dot(V.T)
    print("Final T from A and V.T \n", T)
    return B

"""scikit-learn provides a TruncatedSVD class that implements this capability directly"""
def svd_dimensionality_reduction_TruncatedSVD(A, dimension = 2):
    print("********** scikit-learn provides a TruncatedSVD class *************")
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=dimension)
    svd.fit(A)
    #print("svd \n",svd)
    result = svd.transform(A)
    #print("Transform Matrix is \n ",result)
    return result



# define a matrix
A = array([[1, 2], [3, 4], [5, 6]])
print(A)
# SVD : Calculate Singular-Value Decomposition
U, s, V = svd(A)
print("U is \n",U)
print("s is \n",s)
print("V is \n",V)


print("Reconstruct A Matrix with svd  \n",reconstructMatrix(A =A, s = s, U = U , V =V))
print("Pseudo Inverse A Matrix with svd  \n", pseudoInverse(A =A, s = s, U = U , V =V))
print("svd dimensionality reduction A Matrix with svd  \n", svd_dimensionality_reduction(A =A, s = s, U = U , V =V))


A = array([
	[1,2,3,4,5,6,7,8,9,10],
	[11,12,13,14,15,16,17,18,19,20],
	[21,22,23,24,25,26,27,28,29,30]])

print("svd dimensionality reduction A Matrix \n", A)
print("svd dimensionality reduction A Matrix with sklearn decomposition for 2 dimension \n", svd_dimensionality_reduction_TruncatedSVD(A =A , dimension = 2))
print("svd dimensionality reduction A Matrix with sklearn decomposition for 3 dimension \n", svd_dimensionality_reduction_TruncatedSVD(A =A , dimension = 3))