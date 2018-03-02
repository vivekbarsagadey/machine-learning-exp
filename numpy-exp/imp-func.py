"""




    Mean Value
    Variance
    Covariance
    Covariance Matrix


"""

def mean_example():
    '''
    mu = sum(x1, x2, x3, ..., xn) . 1/n
    '''
    print("************ mean_example *********************** \n")
    from numpy import array
    from numpy import mean
    v = array([1,2,3,4,5,6])
    print(v)
    result = mean(v)
    print("mean value is \n", result)

    M = array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
    print("Multi dimention array \n ",M)
    col_mean = mean(M, axis=0)
    print(" Mean of col is ",col_mean)
    row_mean = mean(M, axis=1)
    print(" Mean of row is ",row_mean)


def variance_example():
    '''
    variance of some random variable X is a measure of
    how much values in the distribution vary on average with respect to the mean.
    sigma^2 = sum from 1 to n ( (xi - mu)^2 ) . 1 / (n - 1)
    '''
    print("************ variance_example *********************** \n")
    from numpy import array
    from numpy import var
    v = array([1, 2, 3, 4, 5, 6])
    print(v)
    result = var(v, ddof=1)
    print(result)

    M = array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
    print("Multi dimention array \n ", M)

    col_var = var(M, ddof=1, axis=0)
    print("Variance of col for dimention array \n ", col_var)
    row_var = var(M, ddof=1, axis=1)
    print("Variance of roe for dimention array \n ",row_var)


def std_example():
    '''
    standard deviation is calculated as the square root of the variance
    s = sqrt(sigma^2)
    '''
    print("************ std_example *********************** \n")
    from numpy import array
    from numpy import std
    M = array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
    print(M)
    col_std = std(M, ddof=1, axis=0)
    print("STD of col for dimention array \n ",col_std)
    row_std = std(M, ddof=1, axis=1)
    print("STD of col for dimention array \n ",row_std)


def covariance_example():
    '''
    covariance is the measure of the joint probability for two random variables
    cov(X, Y) = sum (x - E[X]) * (y - E[Y]) * 1/(n - 1)
    '''
    print("************ covariance_example *********************** \n")
    from numpy import array
    from numpy import cov
    x = array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(x)
    y = array([9, 8, 7, 6, 5, 4, 3, 2, 1])
    print(y)
    Sigma = cov(x, y)[0, 1]
    print("covariance value is ",Sigma)

    from numpy import corrcoef
    Sigma = corrcoef(x, y)
    print("covariance value is ",Sigma)


def covariance_matrix_example():
    '''
    The covariance matrix is a square and symmetric matrix that describes the covariance between two or more random variables.
    Sigma(ij) = cov(Xi, Xj)
    '''
    print("************ covariance_matrix_example *********************** \n")
    from numpy import array
    from numpy import cov
    x = array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(x)
    y = array([9, 8, 7, 6, 5, 4, 3, 2, 1])
    print(y)
    Sigma = cov(x, y)
    print("covariance value is ", Sigma)

mean_example()
variance_example()
std_example()
covariance_example()
covariance_matrix_example()