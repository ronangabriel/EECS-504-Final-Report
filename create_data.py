import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
from numpy import genfromtxt

def create_data():
    shapes = (256, 256)

    # Add noise to original Neoscan images
    path = 'Scan Data Y CSV/Step1_AMP.csv'
    data = genfromtxt(path, delimiter=',')

    # 100 test images for fast scan
    numTestA = 100
    path1 = 'Scan Data Y CSV/Fast1_AMP.csv'
    data1 = genfromtxt(path1, delimiter=',')

    path2 = 'Scan Data Y CSV/Fast2_AMP.csv'
    data2 = genfromtxt(path2, delimiter=',')

    stddev = np.std(data1 - data2) / 3
    filenameOrig = 'fastScanAmpTest'
    for i in range(numTestA):
        gauss = np.random.normal(0, stddev, data1.shape)
        img = data1 + gauss
        img = cv2.resize(img, shapes)
        filename = filenameOrig + str(i)
        with open('fast2step/testA/' + filename + '.csv', 'w') as csvfile: 
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(img)

    # 100 test images for step scan
    numTestB = 100
    path1 = 'Scan Data Y CSV/Step1_AMP.csv'
    data1 = genfromtxt(path1, delimiter=',')

    path2 = 'Scan Data Y CSV/Step2_AMP.csv'
    data2 = genfromtxt(path2, delimiter=',')

    stddev = np.std(data1 - data2) / 3
    filenameOrig = 'stepScanAmpTest'
    for i in range(numTestB):
        gauss = np.random.normal(0, stddev, data.shape)
        img = data1 + gauss
        img = cv2.resize(img, shapes)
        filename = filenameOrig + str(i)
        with open('fast2step/testB/' + filename + '.csv', 'w') as csvfile: 
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(img)

    # train images for fast scan
    numTrainA = 400
    path1 = 'Scan Data Y CSV/Fast2_AMP.csv'
    data1 = genfromtxt(path1, delimiter=',')

    path2 = 'Scan Data Y CSV/Fast3_AMP.csv'
    data2 = genfromtxt(path2, delimiter=',')

    path3 = 'Scan Data Y CSV/Fast4_AMP.csv'
    data3 = genfromtxt(path3, delimiter=',')

    path4 = 'Scan Data Y CSV/Fast5_AMP.csv'
    data4 = genfromtxt(path4, delimiter=',')

    stddev = np.std(data1 - data2) / 3
    filenameOrig = 'fastScanAmpTrain'
    for i in range(numTrainA):
        img = None
        gauss = np.random.normal(0, stddev, data1.shape)
        if 0 <= i and i < 100:
            img = data1 + gauss
        elif i <= 100 and i < 200:
            img = data2 + gauss
        elif i <= 200 and i < 300:
            img = data3 + gauss
        else:
            img = data4 + gauss

        img = cv2.resize(img, shapes)
        filename = filenameOrig + str(i)
        with open('fast2step/trainA/' + filename + '.csv', 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(img)

    # train images for step scan
    numTrainB = numTrainA
    path1 = 'Scan Data Y CSV/Step2_AMP.csv'
    data1 = genfromtxt(path1, delimiter=',')

    path2 = 'Scan Data Y CSV/Step3_AMP.csv'
    data2 = genfromtxt(path2, delimiter=',')

    path3 = 'Scan Data Y CSV/Step4_AMP.csv'
    data3 = genfromtxt(path3, delimiter=',')

    path4 = 'Scan Data Y CSV/Step5_AMP.csv'
    data4 = genfromtxt(path4, delimiter=',')

  
    stddev = np.std(data1 - data2) / 3
    filenameOrig = 'stepScanAmpTrain'
    for i in range(numTrainB):
        img = None
        gauss = np.random.normal(0, stddev, data.shape)
        if 0 <= i and i < 100:
            img = data1 + gauss
        elif i <= 100 and i < 200:
            img = data2 + gauss
        elif i <= 200 and i < 300:
            img = data3 + gauss
        else:
            img = data4 + gauss

        img = cv2.resize(img, shapes)
        filename = filenameOrig + str(i)
        with open('fast2step/trainB/' + filename + '.csv', 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(img)

    '''
    path = 'Scan Data Y CSV/Step1_AMP.csv'
    data = genfromtxt(path, delimiter=',')
    
    path = 'Scan Data Y CSV/Step5_AMP.csv'
    data2 = genfromtxt(path, delimiter=',')
    
    path = 'Scan Data Y CSV/Fast1_AMP.csv'
    data3 = genfromtxt(path, delimiter=',')

    path = 'Scan Data Y CSV/Fast5_AMP.csv'
    data4 = genfromtxt(path, delimiter=',')


    diff = abs(data - data2)
    avg = np.mean(diff)
    stddev = np.std(data - data2) / 5

    diff2 = abs(data3 - data4)
    avg2 = np.mean(diff2)
    stddev2 = np.std(data3 - data4) / 5

    gauss = np.random.normal(0, stddev, data.shape)
    gauss2 = np.random.normal(0, stddev2, data3.shape)

    test = data3 + gauss2
    testNew = cv2.resize(test, data.shape)

    step = data + gauss

    img = plt.imshow(step, cmap='jet')
    plt.colorbar(img)
    plt.show()

    img2 = plt.imshow(test, cmap='jet')
    plt.colorbar(img2)
    plt.show()

    img = plt.imshow(diff, cmap='jet')
    plt.colorbar(img)
    plt.show()

    '''

    print('done')
    return

def main():
    create_data()
    return

if __name__ == "__main__":
    main()