import numpy as np
import sys
import random

def find_closest_k(i, k):
    closest = np.empty((0, 2))

    for j in range(SIZE_OF_TRAIN):
        dist = np.linalg.norm(test_x[i] - train_x[j])
        ##dist = np.sqrt(np.power(test_x[i][0] - train_y[j][0], 2) + np.power(test_x[i][1] - train_y[j][1], 2) + np.power(test_x[i][2] - train_y[j][2], 2) +
                       ##np.power(test_x[i][3] - train_y[j][3], 2) + np.power(test_x[i][4] - train_y[j][4], 2))

        if np.size(closest, 0) == k:
            #furthest, index = None, None
            if dist < closest[k-1][0]:
                closest[k-1][0], closest[k-1][1] = dist, j
            """for d in reversed(range(k)):
                if dist < closest[d][0]:
                    closest[d][0], closest[d][1] = dist, j
                    break
                    if furthest is not None and most_close[d][0] > furthest:
                        furthest, index = most_close[d][0], d
                    elif furthest is None:
                        furthest, index = most_close[d][0], d
            if furthest is not None:
                most_close[d][0], most_close[d][1] = dist, j"""

        else:
            closest = np.concatenate((closest, np.array([[dist, j]])))

        closest = closest[closest[:, 0].argsort()]
    return closest


def test_knn():
    k = 9
    labels = np.empty((SIZE_OF_TEST, 1))
    for i in range(SIZE_OF_TEST):
        k_closest = find_closest_k(i, k)
        #print(f"------- {i} --------")
        #print(k_closest)

        classes = np.empty(k, dtype=np.int64)
        for j in range(k):
            classes[j] = int(train_y[int(k_closest[j, 1])])
        labels[i] = np.bincount(classes).argmax()
    return labels




##############################
# from files
train_x_fname, train_y_fname, test_x_fname, out_fname = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
train_x = np.loadtxt(train_x_fname, delimiter=',')
train_y = np.loadtxt(train_y_fname, delimiter=',')
test_x = np.loadtxt(test_x_fname, delimiter=',')
# sizes
SIZE_OF_TRAIN = np.size(train_x, 0)
SIZE_OF_TEST = np.size(test_x, 0)
NUM_OF_FEATURE = np.size(train_x, 1)
##############################


########## VALIDATION KNN ##################


"""VAL_SIZE = 24

validation_x, validation_y = np.empty((VAL_SIZE, 5)), np.empty((VAL_SIZE, 1))
a = random.sample(range(240), VAL_SIZE)
a.sort()
for index in range(VAL_SIZE):
    validation_x[index], validation_y[index] = train_x[a[index]], train_y[a[index]]

for l in reversed(range(VAL_SIZE)):
    print(a[l])
    train_x = np.delete(train_x, a[l], axis=0)
    train_y = np.delete(train_y, a[l])

SIZE_OF_TRAIN = SIZE_OF_TRAIN - VAL_SIZE
SIZE_OF_TEST = VAL_SIZE

test_x = validation_x"""


#print("hello")


labels_knn = test_knn()

"""for i in range(VAL_SIZE):
    print(str(labels_knn[i]) + "      " + str(validation_y[i]))"""

f = open(out_fname, "w")

for w in range(SIZE_OF_TEST):
    knn_yhat = int(labels_knn[w])
    #print(g)
    f.write(f"knn: {knn_yhat}, perceptron: {0}, svm: {0}, pa: {0}\n")

f.close()