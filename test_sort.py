import numpy as np

#import random
#a = random.sample(range(240), 24)
#print(a)

"""a = np.arange(12).reshape(3, 4)
print(a)

a = np.delete(a, 1, 0)
print(a)"""

train_y = np.loadtxt("train_y.txt", delimiter=',', dtype=np.int64)
#arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
s1 = np.split(train_y, 10)
print(s1)

train_x = np.loadtxt("train_x.txt", delimiter=',')
s2 = np.split(train_x, 10)
print(s2)


"""arr = np.array([1,3,6])
arr2 = np.array([7,4,3])
print(arr)
arr = arr + arr2
print(arr)"""

#arr = np.array([[50, 49, 2], [0, 1, 5]])

#arr = np.array([[50, 0], [51, 1], [2, 5]])

#print(arr)
#arr.sort(axis=0)

#arr = arr[arr[:, 0].argsort()]

#print(arr)

#for i in reversed(range(3)):
#    print(arr[i])