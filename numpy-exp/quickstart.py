import numpy as np
import math as math

a = np.arange(15).reshape(1,15)
print("np.arange(15).reshape(1,15) >>> ",a)
a = np.arange(15).reshape(3, 5)
print("np.arange(15).reshape(3, 5) >>> ",a)
print("shape >>> ",a.shape)
print("ndim >>> ",a.ndim)
print("dtype >>> ",a.dtype.name)
print("itemsize >>> ",a.itemsize)
print("size >>> ",a.size)


"""
Array Creation
"""

print("================ Array Creation ===============")

a = np.array([2,3,4],dtype="float64")
print("Array  >>> ",a)

print(np.array([1, 2, 3.0]))
print(np.array([[1, 2], [3, 4]]))
print(np.array([1, 2, 3], ndmin=2))
print("complex (1d) >>> ",np.array([1, 2, 3], dtype=complex))
print("complex (2d) >>> ",np.array( [ [1,2], [3,4] ], dtype=complex ))

"""
transforms sequences of sequences
"""

print("================ transforms sequences of sequences ===============")
b = np.array([(1.5, 2, 3), (4, 5, 6)])
print("transforms sequences of sequences >> ",b)

"""
Default values
"""
print("================ Default values ===============")

print("Default values of zeros  >> ",np.zeros( (3,4) ))
print("Default values of ones  >> ",np.ones( (3,4) ))
print("Default values of empty  >> ",np.empty( (3,4) ))

"""
range 
"""

print("================ Range values ===============")

print("ranges ",np.arange( 10, 30, 5 ))
print("ranges ",np.arange( 0, 2, 0.3  ))
print("ranges (9 equal div) ",np.linspace( 0, 2, 9 ) )


"""
Basic operation
"""

print("================ Basic operation ===============")

a = np.array( [20,30,40,50] )
b = np.arange( 4 )
print("a",a)
print("b",b)
print("a+b",a+b)
print("a-b",a-b)
print("a*b",a*b)
#print("a/b",a/b)
print("np.sin(a) ",np.sin(a))

A = np.array( [[1,1],
            [0,1]] )
B = np.array( [[2,0],
            [3,4]] )

print("a   >>> ",A)
print("b   >>> ",B)
print("a+b >>> ",A+B)
print("a-b >>> ",A-B)
print("a*b >>> ",A*B)
print("a,dot(b) >>> ",A.dot(B))

a = np.ones((2,3), dtype=int)
b = np.random.random((2,3))
print("3*a" , 3*a)
print("random b (2,3) size >>> " , b)
print("linspace>>> " , np.linspace(0,math.pi,3))
print("np.exp(a*1j) ",np.exp(a*1j))


a = np.random.random((2,3))
print("a.sum() ",a.sum())
print("a.min() ",a.min())
print("a.max() ",a.max())


b = np.arange(12).reshape(3,4)
print("b ",b )
print("b.sum() ",b.sum() )
print("b.sum(axis=0)  ",b.sum(axis=0) )
print("b.sum(axis=1)  ",b.sum(axis=1) )


B = np.arange(3)
print("B  ",B )


print("np.exp(B)",np.exp(B))

print("np.sqrt(B)",np.sqrt(B))

C = np.array([2., -1., 4.])
print('np.add(B, C)',np.add(B, C))

"""
Indexing, Slicing and Iterating
"""

print("================ Indexing, Slicing and Iterating ===============")

a = np.arange(10)**3
print('a',a)
print('a[2]',a[2])
print('a[2:5]',a[2:5])
print('a[:6:2]',a[:6:2])
print('a[ : :-1] (# reversed a) ',a[ : :-1] )

def f(x,y):
    return 10*x+y

b = np.fromfunction(f,(5,4),dtype=int)
for element in b.flat:
    print(element)


print('b',b)

"""
Shape Manipulation
"""
print("=============== Shape Manipulation =========================")
a = np.floor(10*np.random.random((3,4)))
print('a.shape' , a.shape)
print('a' , a)
print('a.ravel()' , a.ravel())
print("a.reshape(6,2)",a.reshape(6,2))
print("a.T convert col to row and row to col >>> ",a.reshape(6,2).T)

"""
Stacking together different arrays
"""
print("============ Stacking together different arrays =====================")
a = np.floor(10*np.random.random((2,2)))
print("a = np.floor(10*np.random.random((2,2)))" ,a)
b = np.floor(10*np.random.random((2,2)))
print("b = np.floor(10*np.random.random((2,2)))" ,b)

print("np.vstack((a,b))" ,np.vstack((a,b)))
print("np.hstack((a,b))" ,np.hstack((a,b)))

"""
Splitting one array into several smaller ones
"""
print("========== Splitting one array into several smaller ones ================")
a = np.floor(10*np.random.random((2,12)))
print(a)
print(np.hsplit(a,3))   # Split a into 3
print(np.hsplit(a,(3,4)))   # Split a after the third and the fourth column



"""
Copies and Views
"""
print("========== Copies and Views ================")

a = np.arange(12)
print("a.shape ",a.shape)
b = a
print("b is a ",b is a)
b.shape = 3, 4 # changes the shape of a
print("a.shape ",a.shape)
print("id of a  : " , id(a))
c = a.view()
print("c is a" , c is a)
print("c.base is a" , c.base is a)
print("c.flags.owndata is a" , c.flags.owndata)
c.shape = 2,6
print("a.shape ",a.shape)
c[0,4] = 1234
print("a >>> ",a)
s = a[ : , 1:3]
print("s >>> ",s)
s[:] = 10
print(s)

#Deep Copy
d = a.copy()

"""
Linear Algebra
"""
print("========== Linear Algebra ================")
a = np.array([[1.0, 2.0], [3.0, 4.0]])
print(a)
print("a.transpose()" , a.transpose())
print("np.linalg.inv(a)", np.linalg.inv(a))
print("np.trace(a)", np.trace(a))





