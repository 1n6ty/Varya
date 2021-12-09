import numpy as np
from tensorflow.keras.datasets import mnist
from functools import reduce

def conv(arr: np.ndarray, filter: np.ndarray) -> np.ndarray:
    res = np.zeros((len(arr) - len(filter) + 1, len(arr[0]) - len(filter[0]) + 1))
    for i in range(len(res)):
        for j in range(len(res[0])):
            res[i][j] = np.sum(arr[i: i + len(filter), j: j + len(filter[0])] * filter)
    return res

def fullConv(arr: np.ndarray, filter: np.ndarray) -> np.ndarray:
    topBottomPad = len(filter) - 1
    leftRightPad = len(filter[0]) - 1
    arr = np.pad(arr, ((topBottomPad, topBottomPad), (leftRightPad, leftRightPad)), 'constant')
    res = np.zeros((len(arr) - len(filter) + 1, len(arr[0]) - len(filter[0]) + 1))
    for i in range(len(arr) - len(filter), -1, -1):
        for j in range(len(arr[0]) - len(filter[0]), -1, -1):
            res[len(res) - i - 1][len(res[0]) - j - 1] = np.sum(arr[i: i + len(filter), j: j + len(filter)] * filter)
    return res

def pool(arr: np.ndarray, shapePool: tuple):
    res = np.zeros_like(arr)
    newArr = np.zeros((int(len(arr) / shapePool[0]), int(len(arr[0]) / shapePool[1])))
    newArrI, newArrJ = 0, 0
    for i in range(0, len(arr), shapePool[0]):
        newArrJ = 0
        for j in range(0, len(arr[0]), shapePool[1]):
            subArr = arr[i: i + shapePool[0], j: j + shapePool[1]]
            maxIndex = subArr.argmax()
            x, y = maxIndex % shapePool[0] + j, maxIndex // shapePool[1] + i
            res[y][x] = 1
            newArr[newArrI][newArrJ] = np.max(subArr)
            newArrJ += 1
        newArrI += 1
    return [res, newArr]

def sigmoid(x, d = False):
    return 1 / (1 + np.exp(-x)) if not d else sigmoid(x) * (1 - sigmoid(x))

def softmax(arr, d = False):
    if not d:
        e = np.exp(arr)
        return e / e.sum()
    else:
        grad = np.zeros((arr.shape[1], arr.shape[1]))
        for i in range(arr.shape[1]):
            for j in range(arr.shape[1]):
                if i == j:
                    grad[i][j] = arr[0][i] * (1 - arr[0][j])
                else:
                    grad[i][j] = -arr[0][i] * arr[0][j]
        return np.sum(grad, axis=0)

def getNum(y):
    res = np.zeros((1, 10))
    res[0][y] = 1
    return res

def unPool(arr, size):
    arr = np.repeat(arr, [size for i in range(arr.shape[0])], axis=0)
    return np.repeat(arr, [size for i in range(arr.shape[1])], axis=-1)

def mltUnPool(arr, size):
    return np.array([unPool(i, size) for i in arr])

(x_train, y_train), (x_test, y_test) = mnist.load_data()

fst_sixfilters = [[np.random.randint(-1, 2, (5, 5)) for j in range(1)] for i in range(6)]
sec_sixteenfilters = [[np.random.randint(-1, 2, (5, 5)) for j in range(6)] for i in range(16)]
thr_onehtwentyfilters = [[np.random.randint(-1, 2, (5, 5)) for j in range(16)] for i in range(120)]

fullcon1Weights = (np.random.random((84, 120)) * 2 - 1) * 0.5
fullcon2Weights = (np.random.random((10, 84)) * 2 - 1) * 0.5

a = 0.01

for ind in range(1000):
    #forwardprop
    inp = np.pad(x_train[ind] / 255, ((2, 2), (2, 2)), 'constant')
    C1Sum = [conv(inp, fst_sixfilters[i][0]) for i in range(6)]
    C1 = [sigmoid(i) for i in C1Sum]

    S2 = [pool(C1[i], (2, 2)) for i in range(6)]
    S2_layers = [S2[i][1] for i in range(6)]
    S2_marks = [S2[i][0] for i in range(6)]

    C3Sum = [reduce(lambda x, y: x + y, [conv(S2_layers[i], sec_sixteenfilters[j][i]) for i in range(6)], np.zeros((10, 10))) + 1 for j in range(16)]
    C3 = [sigmoid(i) for i in C3Sum]
    
    S4 = [pool(C3[i], (2, 2)) for i in range(16)]
    S4_layers = [S4[i][1] for i in range(16)]
    S4_marks = [S4[i][0] for i in range(16)]

    C5Sum = np.array([reduce(lambda x, y: x + y, [conv(S4_layers[i], thr_onehtwentyfilters[j][i]) for i in range(16)], np.zeros((1, 1))) + 1 for j in range(120)])
    C5 = [sigmoid(i) for i in C5Sum]

    fullcon1 = np.array(C5).reshape((1, 120))

    fullcon2Sum = fullcon1 @ fullcon1Weights.reshape((120, 84))
    fullcon2 = sigmoid(fullcon2Sum)

    outSum = fullcon2 @ fullcon2Weights.reshape((84, 10))
    out = softmax(outSum)

    loss = out - getNum(y_train[ind]) #RMSE
    
    outloss = loss * np.sum(softmax(outSum, d = True), axis=0)
    print(np.sum(np.square(getNum(y_train[ind]) - out)) / 11)

    fullcon2Loss = (outloss @ fullcon2Weights) * sigmoid(fullcon2Sum, d = True)
    fullcon2Weights -= (outloss.reshape((10, 1)) @ fullcon2) * a

    fullcon1loss = (fullcon2Loss @ fullcon1Weights)
    fullcon1Weights -= (fullcon2Loss.reshape((84, 1)) @ fullcon1) * a

    C5loss = [np.array(i).reshape(1, 1) for i in fullcon1loss[0]]
    C5loss = [C5loss[i] * sigmoid(C5Sum[i], d = True) for i in range(120)]

    S4loss = [reduce(lambda x, y: x + y, [fullConv(np.rot90(thr_onehtwentyfilters[j][i], 2), C5loss[j]) for j in range(120)], np.zeros((5, 5))) for i in range(16)]
    thr_onehtwentyfilters = [[thr_onehtwentyfilters[i][j] - conv(S4_layers[j], C5loss[i]) * a for j in range(16)] for i in range(120)]

    C3loss = [unPool(S4loss[i], 2) * S4_marks[i] for i in range(16)]
    C3loss = [C3loss[i] * sigmoid(C3Sum[i], d = True) for i in range(16)]

    S2loss = [reduce(lambda x, y: x + y, [fullConv(np.rot90(sec_sixteenfilters[j][i], 2), C3loss[j]) for j in range(16)], np.zeros((14, 14))) for i in range(6)]
    sec_sixteenfilters = [[sec_sixteenfilters[i][j] - conv(S2_layers[j], C3loss[i]) * a for j in range(6)] for i in range(16)]

    C1loss = [unPool(S2loss[i], 2) * S2_marks[i] for i in range(6)]
    C1loss = [C1loss[i] * sigmoid(C1Sum[i], d = True) for i in range(6)]

    fst_sixfilters = [[fst_sixfilters[i][j] - conv(inp, C1loss[i]) * a for j in range(1)] for i in range(6)]
    print(out, y_train[ind], fst_sixfilters[0])