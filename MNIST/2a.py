import numpy as np
import struct
from array import array
import matplotlib.pyplot as plt
from os.path  import join

def runTest():

    trainImageFile = 'train-images.idx3-ubyte'
    trainLabelFile = 'train-labels.idx1-ubyte'
    trainImages, trainLabels = readImagesAndLabels(trainLabelFile, trainImageFile)
    trainImagesSorted = parseImagesIntoArrays(trainImages, trainLabels)
    
    pcMeans = []
    stds = []
    for digit in range(len(trainImagesSorted)):
        mat = np.array(trainImagesSorted[digit])
        mat = mat.reshape(len(trainImagesSorted[digit]) ,784)
        pcmean, std = meansAndStds(mat)
        pcMeans.append(pcmean)
        stds.append(std)
    digitNum = 0
    for digit in pcMeans:
        workableArr = np.reshape(digit, (28,28))
        plt.figure()
        plt.title("Mean of each pixel of {}".format(digitNum))
        plt.imshow(workableArr, interpolation = "none", cmap = "gray")
        plt.show()
        digitNum += 1
    digitNum = 0
    for digit in stds:
        workableArr = np.reshape(digit, (28,28))
        plt.figure()
        plt.title("Standard Deviation of {}".format(digitNum))
        plt.imshow(workableArr, interpolation = "none", cmap = "gray")
        plt.show()
        digitNum += 1
            
def readImagesAndLabels(labelFile, imageFile):
    path = '..\\'
    labelFile = join(path, labelFile)
    imageFile = join(path, imageFile)
    labels = []
    with open(labelFile,'rb') as l:
        magic, size = struct.unpack(">II", l.read(8))
        if magic != 2049:
            raise ValueError('Magic number error, expected 2049, got {}'.format(magic))
        labels = array("B", l.read()) 
        
    with open(imageFile,'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        if magic != 2051:
            raise ValueError('Magic number error, expected 2051, got {}'.format(magic))
        image_data = array("B", f.read())
        images = []
    for i in range(size):
        images.append([0]*nrows*ncols)
    for i in range(size):
        img = np.array(image_data[i*nrows*ncols:(i+1)*nrows*ncols])
        img = img.reshape(1, 784)
        images[i][:] = img
    return images, labels
    
def parseImagesIntoArrays(images, labels):
    arrOfImagesSortedByLabel = {}
    for i in range(len(images)):
        if labels[i] in arrOfImagesSortedByLabel:
            arrOfImagesSortedByLabel[labels[i]].append(np.matrix(images[i]))
        else:
            arrOfImagesSortedByLabel[labels[i]] = [np.matrix(images[i])]
    return arrOfImagesSortedByLabel
        
def meansAndStds(matrix):
    means = []
    stds = []
    for dig in matrix.T:
        means.append(np.mean(dig))
        stds.append(np.std(dig))
    return means, stds

if __name__ == "__main__":
    runTest()