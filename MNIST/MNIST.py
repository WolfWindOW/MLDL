# -*- coding: utf-8 -*-
import numpy as np
from numpy import linalg as la
import struct
from array import array
import matplotlib.pyplot as plt
from os.path  import join
#from scipy.linalg import eigh
#from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
#from sklearn import decomposition
from math import sqrt
np.random.seed(2)

#Constants for PCA
D = 5
M = 3
K = 10
dimRed = [1, 2, 8, 16, 32, 64, 128, 256, 612, 783]

def runTest():
    #testImageFile = 't10k-images.idx3-ubyte'
    #testLabelFile = 't10k-labels.idx1-ubyte'
    trainImageFile = 'train-images.idx3-ubyte'
    trainLabelFile = 'train-labels.idx1-ubyte'
    trainImages, trainLabels = readImagesAndLabels(trainLabelFile, trainImageFile)
    #testImages, testLabels = readImagesAndLabels(testLabelFile, testImageFile)
    trainImagesSorted = parseImagesIntoArrays(trainImages, trainLabels)    
    del trainImageFile, trainLabelFile, trainImages, trainLabels
    #for digit in range(len(trainImagesSorted)):
    #    mat = np.array(trainImagesSorted[digit])
    #    mat = mat.reshape(len(trainImagesSorted[digit]) ,784)
    #   pc, pcmean = pcaViaSVD(mat, K)
    #    principal_C.append(pc)
    #    pcMeans.append(pcmean)
    #digitNum = 0
    #for digit in pcMeans:
    #    #workableArr = np.reshape(digit, (28,28))
    #    plt.figure()
    #    plt.title("Mean of each PC, {}".format(digitNum))
    #    plt.xlabel("PC Number")
    #    plt.ylabel("Value")
    #    plt.plot(digit)
    #    plt.show()
    #    digitNum += 1
    nrmsesTotal = []
    for k in range(len(dimRed)):
        #principal_C = []
        #pcMeans = []
        #r2s = []
        nrmses = []
        for digit in range(len(trainImagesSorted)):
            mat = np.array(trainImagesSorted[digit])
            mat = mat.reshape(len(trainImagesSorted[digit]),784)
            pc, pcmean, recon = pcaViaSVD(mat, dimRed[k])
            #principal_C.append(pc)
            #pcMeans.append(pcmean)
            #r2 = 0
            #r2 += r2_score(mat, recon)
            rmse = sqrt(mean_squared_error(mat, recon))
            nrmse = rmse/sqrt(np.mean(mat**2))
            #r2s.append(r2)
            nrmses.append(nrmse)
        nrmsesTotal.append(nrmses)
    for i in range(10):
        nrmseDigit = []
        for n in range(len(nrmsesTotal)):
            nrmseDigit.append(nrmsesTotal[n][i]) 
        plt.figure()
        plt.title("Normalized Root Mean Square Error for Digit {}".format(i))
        default_x_ticks = range(len(dimRed))
        plt.xticks(default_x_ticks, dimRed)
        plt.xlabel("PC number")
        plt.ylabel("R2")
        plt.plot(default_x_ticks, nrmseDigit)
        plt.show()
        
    
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
        
def pcaViaSVD(matrix, k):
    #C = np.matmul(matrix.T, matrix)
    #print("C = \n", C)
    #l, principalA = eigh(C, eigvals=((784-K, 783)))
    #idx = l.argsort()[::-1]
    #l, principalA = l[idx], principalA[:, idx]
    #principalA = principalA.T
    #newCoords = np.matmul(principalA, matrix.T)
    #print("l = \n", l)
    #print("V = \n", principalA)
    #principalC = matrix.dot(principalA)
    #print("Y = \n", principalC)
    U, s, Vt = la.svd(matrix, full_matrices=False)
    #V = Vt.T
    S = np.diag(s)
    #PC_k = principalC[:, 0:K]
    US_k = U[:, 0:k].dot(S[0:k, 0:k])
    recon = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    means = []
    for pc in US_k.T:
        minVal = np.min(pc)
        for e in range(len(pc)):
            val = pc[e]
            newVal = (val-minVal)
            pc[e] = newVal
        maxVal = np.max(pc)
        for e in range(len(pc)):
            val = pc[e]
            newVal = ((val)/maxVal)
            pc[e] = newVal
        #means.append(np.mean(pc))

    return US_k, means, recon
    
if __name__ == "__main__":
    runTest()
    