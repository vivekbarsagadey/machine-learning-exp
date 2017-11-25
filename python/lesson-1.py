"""

Iterators

"""

nums = [1, 2, 3]
print("iter  :   {} ".format(iter(nums)))
print("iter  :   {} ".format(nums.__iter__()))
print("reversed  :   {} ".format(nums.__reversed__() ))

it = iter(nums)
print("next(it)  :   {} ".format(next(it)))
print("next(it)  :   {} ".format(next(it)))
print("next(it)  :   {} ".format(next(it)))




def f():
  print("-- start --")
  yield 1
  print("-- middle --")
  yield 2
  print("-- finished --")
gen = f()

next(gen)
next(gen)
#next(gen)

