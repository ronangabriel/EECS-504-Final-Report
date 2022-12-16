import numpy as np
import matplotlib.pyplot as plt
import csv
import tensorflow as tf
import random


def read_data():
    arr = []
    filename = 'Fast5_AMP'
    with open('Scan Data Y/' + filename + '.dat') as f:
        lines = f.readlines()
        for line in lines:
            arr.append(line.split('\t'))
        arr = np.array(arr).astype(dtype=float)
    f.close()

    arr = np.flip(arr, axis=0)
    img = plt.imshow(arr, cmap='jet')
    plt.colorbar(img)
    plt.show()

    with open('Scan Data Y CSV/' + filename + '.csv', 'w') as csvfile: 
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(arr)

    return

def read_data_tensor():
    testA = []
    testB = []
    trainA = []
    trainB = []

    numTest = 100
    numTrain = 400

    for i in range(numTest):
        print(str(i) + '/' + str(numTest))
        path = open('fast2step/testA/fastScanAmpTest' + str(i) + '.csv')
        arr = np.loadtxt(path, delimiter=",",dtype='float32')
        arr = np.reshape(arr, (256, 256, 1))
        #arr = np.repeat(arr, repeats=3, axis=2)
        arr = tf.convert_to_tensor(arr)
        testA.append(arr)

        path = open('fast2step/testB/stepScanAmpTest' + str(i) + '.csv')
        arr = np.loadtxt(path, delimiter=",",dtype='float')
        arr = np.reshape(arr, (256, 256, 1))
        #arr = np.repeat(arr, repeats=3, axis=2)
        arr = tf.convert_to_tensor(arr)
        testB.append(arr)

    for i in range(numTrain):
        print(str(i) + '/' + str(numTrain))
        path = open('fast2step/trainA/fastScanAmpTrain' + str(i) + '.csv')
        arr = np.loadtxt(path, delimiter=",",dtype='float32')
        arr = np.reshape(arr, (256, 256, 1))
        #arr = np.repeat(arr, repeats=3, axis=2)
        arr = tf.convert_to_tensor(arr)
        trainA.append(arr)

        path = open('fast2step/trainB/stepScanAmpTrain' + str(i) + '.csv')
        arr = np.loadtxt(path, delimiter=",",dtype='float32')
        arr = np.reshape(arr, (256, 256, 1))
        #arr = np.repeat(arr, repeats=3, axis=2)
        arr = tf.convert_to_tensor(arr)
        trainB.append(arr)

    #randomize the order
    random.seed(3)
    random.shuffle(testA)
    random.shuffle(testB)
    random.shuffle(trainA)
    random.shuffle(trainB)

    testA = tf.stack(testA)
    testB = tf.stack(testB)
    trainA = tf.stack(trainA)
    trainB = tf.stack(trainB)

    return testA, testB, trainA, trainB

def main():
    a, b, c, d = read_data_tensor()
    return

if __name__ == "__main__":
    main()