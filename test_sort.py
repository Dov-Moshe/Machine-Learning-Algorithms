import numpy as np

#import random
#a = random.sample(range(240), 24)
#print(a)

a = np.arange(12).reshape(3, 4)
print(a)

a = np.delete(a, 1, 0)
print(a)


#arr = np.array([[50, 49, 2], [0, 1, 5]])

#arr = np.array([[50, 0], [51, 1], [2, 5]])

#print(arr)
#arr.sort(axis=0)

#arr = arr[arr[:, 0].argsort()]

#print(arr)

#for i in reversed(range(3)):
#    print(arr[i])