"""
softmax-numpy

"""

import numpy as np

"""
p(Cn)= exp{θ⋅Xn} / ∑Ni=1 exp{θ⋅Xi}
"""

def exp_1():
    X = np.array([1.1, 5.0, 2.8, 7.3])
    ##print(X)
    theta = 2.0  # determinism parameter

    ps = np.exp(X * theta)
    ps = ps/ np.sum(ps)
    ##print(ps)
    print("Softmax Function Output :: {}".format(ps))





exp_1();
