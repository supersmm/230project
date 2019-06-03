# Functions used in the training and test process. Some of them are collected from Microsoft open source.

from __future__ import print_function
import numpy as np
import pdb


def getConfusionMatrix(predicted, target, numClasses=2):
    '''
    Returns a confusion matrix for a multiclass classification
    problem. `predicted` is a 1-D array of integers representing
    the predicted classes and `target` is the target classes.
    confusion[i][j]: Number of elements of class j
        predicted as class i
    Labels are assumed to be in range(0, numClasses)
    Use`printFormattedConfusionMatrix` to echo the confusion matrix
    in a user friendly form.
    '''
    predicted = np.argmax(predicted, axis=1)
    assert(predicted.ndim == 1)
    assert(target.ndim == 1)
    arr = np.zeros([numClasses, numClasses])

    for i in range(len(predicted)):
        arr[predicted[i]][target[i]] += 1
    return arr


def printFormattedConfusionMatrix(matrix):
    '''
    Given a 2D confusion matrix, prints it in a human readable way.
    The confusion matrix is expected to be a 2D numpy array with
    square dimensions
    '''
    assert(matrix.ndim == 2)
    assert(matrix.shape[0] == matrix.shape[1])
    RECALL = 'Recall'
    PRECISION = 'PRECISION'
    print("|%s|" % ('True->'), end='')
    for i in range(matrix.shape[0]):
        print("%7d|" % i, end='')
    print("%s|" % 'Precision')

    print("|%s|" % ('-' * len(RECALL)), end='')
    for i in range(matrix.shape[0]):
        print("%s|" % ('-' * 7), end='')
    print("%s|" % ('-' * len(PRECISION)))

    precisionlist = np.sum(matrix, axis=1)
    recalllist = np.sum(matrix, axis=0)
    precisionlist = [matrix[i][i] / x if x !=
                     0 else -1 for i, x in enumerate(precisionlist)]
    recalllist = [matrix[i][i] / x if x !=
                  0 else -1 for i, x in enumerate(recalllist)]
    for i in range(matrix.shape[0]):
        # len recall = 6
        print("|%6d|" % (i), end='')
        for j in range(matrix.shape[0]):
            print("%7d|" % (matrix[i][j]), end='')
        print("%s" % (" " * (len(PRECISION) - 7)), end='')
        if precisionlist[i] != -1:
            print("%1.5f|" % precisionlist[i])
        else:
            print("%7s|" % "nan")

    print("|%s|" % ('-' * len(RECALL)), end='')
    for i in range(matrix.shape[0]):
        print("%s|" % ('-' * 7), end='')
    print("%s|" % ('-' * len(PRECISION)))
    print("|%s|" % ('Recall'), end='')

    for i in range(matrix.shape[0]):
        if recalllist[i] != -1:
            print("%1.5f|" % (recalllist[i]), end='')
        else:
            print("%7s|" % "nan", end='')

    print('%s|' % (' ' * len(PRECISION)))


def getPrecisionRecall(cmatrix, label=1):
    trueP = cmatrix[label][label]
    denom = np.sum(cmatrix, axis=0)[label]
    if denom == 0:
        denom = 1
    recall = trueP / denom
    denom = np.sum(cmatrix, axis=1)[label]
    if denom == 0:
        denom = 1
    precision = trueP / denom
    return precision, recall


def getPrecision(predicted, target, numClasses=2):
	return getPrecisionRecall(getConfusionMatrix(predicted, target, numClasses=2))[0]

def getRecall(predicted, target, numClasses=2):
	return getPrecisionRecall(getConfusionMatrix(predicted, target, numClasses=2))[1]


def getMacroPrecisionRecall(cmatrix):
    # TP + FP
    precisionlist = np.sum(cmatrix, axis=1)
    # TP + FN
    recalllist = np.sum(cmatrix, axis=0)
    precisionlist__ = [cmatrix[i][i] / x if x !=
                       0 else 0 for i, x in enumerate(precisionlist)]
    recalllist__ = [cmatrix[i][i] / x if x !=
                    0 else 0 for i, x in enumerate(recalllist)]
    precision = np.sum(precisionlist__)
    precision /= len(precisionlist__)
    recall = np.sum(recalllist__)
    recall /= len(recalllist__)
    return precision, recall


def getMicroPrecisionRecall(cmatrix):
    # TP + FP
    precisionlist = np.sum(cmatrix, axis=1)
    # TP + FN
    recalllist = np.sum(cmatrix, axis=0)
    num = 0.0
    for i in range(len(cmatrix)):
        num += cmatrix[i][i]

    precision = num / np.sum(precisionlist)
    recall = num / np.sum(recalllist)
    return precision, recall


def getMacroMicroFScore(cmatrix):
    '''
    Returns macro and micro f-scores.
    Refer: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.104.8244&rep=rep1&type=pdf
    '''
    precisionlist = np.sum(cmatrix, axis=1)
    recalllist = np.sum(cmatrix, axis=0)
    precisionlist__ = [cmatrix[i][i] / x if x !=
                       0 else 0 for i, x in enumerate(precisionlist)]
    recalllist__ = [cmatrix[i][i] / x if x !=
                    0 else 0 for i, x in enumerate(recalllist)]
    macro = 0.0
    for i in range(len(precisionlist)):
        denom = precisionlist__[i] + recalllist__[i]
        numer = precisionlist__[i] * recalllist__[i] * 2
        if denom == 0:
            denom = 1
        macro += numer / denom
    macro /= len(precisionlist)

    num = 0.0
    for i in range(len(precisionlist)):
        num += cmatrix[i][i]

    denom1 = np.sum(precisionlist)
    denom2 = np.sum(recalllist)
    pi = num / denom1
    rho = num / denom2
    denom = pi + rho
    if denom == 0:
        denom = 1
    micro = 2 * pi * rho / denom
    return macro, micro
