import numpy as np
import sys


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
    # creating matrix of wth and add 1 for bias
    w = np.zeros((NUM_OF_LABELS, NUM_OF_FEATURE+1))
    eta = 0.1
    epochs = 15
    final_w = None
    current_average = 0

    #train_x_after_spl, train_y_after_spl, validation_x, validation_y, val_size = split_train_and_valid(SIZE_OF_TRAIN)

    for t in range(epochs):
        if t == 5:
            eta = 0.01
        elif t == 10:
            eta = 0.001
        # shuffle the train set
        p = np.random.permutation(len(train_x_norm))
        train_x_sh, train_y_sh = train_x_norm[p], train_y[p]

        # one epoch of perceptron algorithm
        for x, y in zip(train_x_sh, train_y_sh):
            x = np.append(x, [1])
            w_dot_x = w.dot(x)
            index_max = np.argmax(w_dot_x)
            if index_max != y:
                w[y] = w[y] + eta * x
                w[index_max] = w[index_max] - eta * x

        """average_validation = validation(w, validation_x, validation_y, val_size)
        if average_validation > current_average:
            # print(average_validation)
            final_w = w
            current_average = average_validation"""

    return w
###########################################


########## PA ALGORITHM ##################
def train_pa():
    # creating matrix of wth and add 1 for bias
    w = np.zeros((NUM_OF_LABELS, NUM_OF_FEATURE + 1))
    epochs = 15
    final_w = None
    current_average = 0

    #train_x_after_spl, train_y_after_spl, validation_x, validation_y = split_train_and_valid(SIZE_OF_TRAIN, 0.2)



    for t in range(epochs):

        # shuffle the train set
        p = np.random.permutation(len(train_x_norm))
        train_x_sh, train_y_sh = train_x_norm[p], train_y[p]

        # for each pair (x,y) in train set
        for x, y in zip(train_x_sh, train_y_sh):
            # appending for bias
            x = np.append(x, [1])
            # evaluating w*x and find the max wth
            w_dot_x = w.dot(x)
            index_max = np.argmax(w_dot_x)
            # getting the w of the right and the wrong labels
            w_right_y = w[y]
            w_wrong_y = w[index_max]
            # calculating tau and updating the wth
            tau = (max(0, 1 - w_right_y.dot(x) + w_wrong_y.dot(x))) / (2 * np.power(np.linalg.norm(x), 2))
            w[y] = w[y] + tau * x
            w[index_max] = w[index_max] - tau * x

        #average_validation = validation(w, validation_x, validation_y, np.size(validation_y, 0))
        #print(f"epochs {t}: " + str(average_validation))
        """if average_validation > current_average:
            print(average_validation)
            final_w = w
            current_average = average_validation"""

    return w


########## SVM ALGORITHM ##################
def train_svm():
    # creating matrix of wth and add 1 for bias
    w = np.zeros((NUM_OF_LABELS, NUM_OF_FEATURE + 1))
    epochs = 15
    eta = 0.1
    gama = 0.3
    final_w = None
    current_average = 0

    #train_x_after_spl, train_y_after_spl, validation_x, validation_y = split_train_and_valid(SIZE_OF_TRAIN)

    for t in range(epochs):
        if t == 5:
            eta = 0.01
        elif t == 10:
            eta = 0.001

        # shuffle the train set
        p = np.random.permutation(len(train_x_norm))
        train_x_sh, train_y_sh = train_x_norm[p], train_y[p]

        # for each pair (x,y) in train set
        for x, y in zip(train_x_sh, train_y_sh):
            # append for bias
            x = np.append(x, [1])
            # evaluating w*x and find the max wth
            w_dot_x = w.dot(x)
            index_max = np.argmax(w_dot_x)
            if index_max != y:
                # getting the w of the right and the wrong labels
                # calculating tau and updating the wth
                w[y] = (1 - eta * gama) * w[y] + eta * x
                w[index_max] = (1 - eta * gama) * w[index_max] - eta * x
                for i in range(NUM_OF_LABELS):
                    if i != y and i != index_max:
                        w[i] = (1 - eta * gama) * w[i]
            else:
                for i in range(NUM_OF_LABELS):
                    w[i] = (1 - eta * gama) * w[i]

        """average_validation = validation(w, validation_x, validation_y, val_size)
        if 100 > average_validation > current_average:
            final_w = w
            current_average = average_validation"""

    return w


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


def split_train_and_valid(train_size, percent):
    # the size of validation set from train set
    val_size = int(train_size * percent)
    validation_x, validation_y = np.empty((val_size, NUM_OF_FEATURE)), np.empty((val_size, 1))
    # getting randomly sample from the train set
    a = np.random.choice(SIZE_OF_TRAIN, size=val_size, replace=False)
    a.sort()
    # crate validation set
    for index in range(val_size):
        validation_x[index], validation_y[index] = train_x_norm[a[index]], train_y[a[index]]
    # removing the validation set from the train set
    for l in reversed(range(val_size)):
        train_x_after_val = np.delete(train_x_norm, a[l], axis=0)
        train_y_after_val = np.delete(train_y, a[l])
    return train_x_after_val, train_y_after_val, validation_x, validation_y
######################


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

# min-max normalization
"""train_x_norm = np.empty((SIZE_OF_TRAIN, NUM_OF_FEATURE))
test_x_norm = np.empty((SIZE_OF_TEST, NUM_OF_FEATURE))
for i in range(NUM_OF_FEATURE):
    old_max, old_min = train_x.max(axis=0)[i], train_x.min(axis=0)[i]
    new_min, new_max = -5, 5
    train_x_norm[:, i] = ((train_x[:, i] - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    test_x_norm[:, i] = ((test_x[:, i] - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min"""

# z-score normalization
train_x_norm = np.zeros((SIZE_OF_TRAIN, NUM_OF_FEATURE))
test_x_norm = np.zeros((SIZE_OF_TEST, NUM_OF_FEATURE))
for i in range(NUM_OF_FEATURE):
    average = np.average(train_x[:, i])
    stand_dev = np.std(train_x[:, i])
    train_x_norm[:, i] = (train_x[:, i] - average) / stand_dev
    test_x_norm[:, i] = (test_x[:, i] - average) / stand_dev


sum_knn = 0
sum_perceptron = 0
sum_svm = 0
sum_pas_ag = 0


for i in range(1):

    # knn test
    labels_knn = predict_test_knn()
    # perceptron train and predict
    w = train_perceptron()
    labels_perceptron = predict(w)
    # svm train and predict
    w = train_svm()
    labels_svm = predict(w)
    # pa train and predict
    w = train_pa()
    labels_pa = predict(w)

    # writing results
    """f = open(out_fname, "w")
    for s in range(SIZE_OF_TEST):
        knn_yhat = int(labels_knn[s])
        perceptron_yhat = int(labels_perceptron[s])
        pa_yhat = int(labels_pa[s])
        svm_yhat = int(labels_svm[s])
        f.write(f"knn: {knn_yhat}, perceptron: {perceptron_yhat}, svm: {svm_yhat}, pa: {pa_yhat}\n")
    f.close()"""


    true_labels = np.loadtxt("test_y.txt", delimiter=',')


    sum_right = 0
    for a, b in zip(true_labels, labels_knn):
        if a == b:
            sum_right = sum_right + 1
    knn_p = (sum_right / np.size(true_labels, 0)) * 100

    #print("knn percent right: " + str(knn_p))

    sum_right = 0
    for a, b in zip(true_labels, labels_perceptron):
        if a == b:
            sum_right = sum_right + 1
    perceptron_p = (sum_right / np.size(true_labels, 0)) * 100

    #print("perceptron percent right: " + str(perceptron_p))

    sum_right = 0
    for a, b in zip(true_labels, labels_svm):
        if a == b:
            sum_right = sum_right + 1
    svm_p = (sum_right / np.size(true_labels, 0)) * 100

    #print("svm percent right: " + str(svm_p))

    sum_right = 0
    for a, b in zip(true_labels, labels_pa):
        if a == b:
            sum_right = sum_right + 1
    pa_p = (sum_right / np.size(true_labels, 0)) * 100

    #print("pa percent right: " + str(pa_p))

    sum_knn = sum_knn + knn_p
    sum_perceptron = sum_perceptron + perceptron_p
    sum_svm = sum_svm + svm_p
    sum_pas_ag = sum_pas_ag + pa_p



print("average knn: " + str(sum_knn / 1))
print("average perceptron: " + str(sum_perceptron / 1))
print("average svm: " + str(sum_svm / 1))
print("average pa: " + str(sum_pas_ag / 1))