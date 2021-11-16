import numpy as np
import sys
import random


########## KNN ALGORITHM ##################
def find_closest_k(i, k):
    closest = np.empty((0, 2))

    for j in range(SIZE_OF_TRAIN):
        dist = np.linalg.norm(test_x[i] - train_x[j])

        if np.size(closest, 0) == k:
            if dist < closest[k-1][0]:
                closest[k-1][0], closest[k-1][1] = dist, j
        else:
            closest = np.concatenate((closest, np.array([[dist, j]])))
        # sorting the current closest sample
        closest = closest[closest[:, 0].argsort()]
    return closest


def predict_test_knn():
    k = 9
    labels = np.empty((SIZE_OF_TEST, 1))
    for i in range(SIZE_OF_TEST):
        # get the closest train sample to the specific test sample
        k_closest = find_closest_k(i, k)

        # get the y value of the closest train sample
        classes = np.empty(k, dtype=np.int64)
        for j in range(k):
            classes[j] = int(train_y[int(k_closest[j, 1])])
        labels[i] = np.bincount(classes).argmax()
    return labels
###########################################


########## PERCEPTRON ALGORITHM ##################
def train_perceptron():

    w = np.zeros((NUM_OF_LABELS, NUM_OF_FEATURE+1))
    #bias = np.zeros((NUM_OF_LABELS, 1))
    final_w = None
    current_average = 0
    eta = 0.001
    epochs = 10
    for v in range(5):
        train_x_after_val, train_y_after_val, validation_x, validation_y, val_size = create_train_and_validation(SIZE_OF_TRAIN)

        for t in range(epochs):
            p = np.random.permutation(len(train_x_after_val))
            train_x_sh, train_y_sh = train_x_after_val[p], train_y_after_val[p]
            for x, y in zip(train_x_sh, train_y_sh):
                x = np.append(x, [1])
                w_dot_x = w.dot(x)
                #w_dot_x = np.add(w.dot(x), bias.T)
                index_max = np.argmax(w_dot_x)
                if index_max != y:
                    w[y] = w[y] + eta * x
                    w[index_max] = w[index_max] - eta * x
                    #bias[y] = bias[y] + eta
                    #bias[index_max] = bias[index_max] - eta

            average_validation = validation(w, validation_x, validation_y, val_size)
            #print(average_validation)
            if 97 > average_validation > current_average:
                final_w = w
                current_average = average_validation
                if current_average > 95:
                    return final_w
    #print(current_average)
    return final_w
    #return w
###########################################


def train_pa():
    w = np.zeros((NUM_OF_LABELS, NUM_OF_FEATURE + 1))
    epochs = 10
    for t in range(epochs):
        p = np.random.permutation(len(train_x_norm))
        train_x_sh, train_y_sh = train_x_norm[p], train_y[p]

        for x, y in zip(train_x_sh, train_y_sh):
            x = np.append(x, [1])
            w_dot_x = w.dot(x)
            index_max = np.argmax(w_dot_x)
            w_right_y = w[y]
            w_wrong_y = w[index_max]
            tau = (max(0, 1 - w_right_y.dot(x) + w_wrong_y.dot(x))) / (2 * np.power(np.linalg.norm(x), 2))
            w[y] = w[y] + tau * x
            w[index_max] = w[index_max] - tau * x

    return w


def predict_pa():
    pass


##### PREDICTION #####
def predict(w):
    labels = np.empty((SIZE_OF_TEST, 1))
    # for every sample in the test set check the predict y
    for index, x in enumerate(test_x_norm):
        w_dot_x = w.dot(np.append(x, [1]))
        index_max = np.argmax(w_dot_x)
        labels[index] = index_max
    # return list of label for the test
    return labels


##### VALIDATION #####
def validation(w, validation_x, validation_y, val_size):
    labels = np.empty((val_size, 1))
    # check the correction of the predict on the validation set
    for index, x in enumerate(validation_x):
        w_dot_x = w.dot(np.append(x, [1]))
        index_max = np.argmax(w_dot_x)
        labels[index] = index_max
    # finding the average of correction
    sum_right = 0
    for a, b in zip(labels, validation_y):
        if a == b:
            sum_right = sum_right + 1
    return (sum_right / val_size) * 100
######################

def create_train_and_validation(train_size):
    # the size of validation set from train set
    val_size = int(train_size / 5)
    validation_x, validation_y = np.empty((val_size, NUM_OF_FEATURE)), np.empty((val_size, 1))
    # getting randomly sample from the train set
    a = random.sample(range(240), val_size)
    a.sort()
    # crate validation set
    for index in range(val_size):
        validation_x[index], validation_y[index] = train_x_norm[a[index]], train_y[a[index]]
    # removing the validation set from the train set
    for l in reversed(range(val_size)):
        train_x_after_val = np.delete(train_x_norm, a[l], axis=0)
        train_y_after_val = np.delete(train_y, a[l])
    return train_x_after_val, train_y_after_val, validation_x, validation_y, val_size


##############################
# from files
train_x_fname, train_y_fname, test_x_fname, out_fname = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
train_x = np.loadtxt(train_x_fname, delimiter=',')
train_y = np.loadtxt(train_y_fname, delimiter=',', dtype=np.int64)
test_x = np.loadtxt(test_x_fname, delimiter=',')
labels_name = np.sort(np.unique(train_y))
# sizes
SIZE_OF_TRAIN = np.size(train_x, 0)
SIZE_OF_TEST = np.size(test_x, 0)
NUM_OF_FEATURE = np.size(train_x, 1)
NUM_OF_LABELS = np.size(labels_name)
##############################

# knn test
labels_knn = predict_test_knn()


# normalization
train_x_norm = np.empty((SIZE_OF_TRAIN, NUM_OF_FEATURE))
test_x_norm = np.empty((SIZE_OF_TEST, NUM_OF_FEATURE))
for i in range(NUM_OF_FEATURE):
    old_max, old_min = train_x.max(axis=0)[i], train_x.min(axis=0)[i]
    new_min, new_max = -5, 5

    for j in range(SIZE_OF_TRAIN):
        train_x_norm[j][i] = ((train_x[j][i] - old_min) / (old_max - old_min))*(new_max - new_min) + new_min
    for j in range(SIZE_OF_TEST):
        test_x_norm[j][i] = ((test_x[j][i] - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min


# perceptron train
w = train_perceptron()
labels_perceptron = predict(w)

w = train_pa()
labels_pa = predict(w)

f = open(out_fname, "w")

for s in range(SIZE_OF_TEST):
    knn_yhat = int(labels_knn[s])
    perceptron_yhat = int(labels_perceptron[s])
    pa_yhat = int(labels_pa[s])
    f.write(f"knn: {knn_yhat}, perceptron: {perceptron_yhat}, svm: {0}, pa: {pa_yhat}\n")

f.close()