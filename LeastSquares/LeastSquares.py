# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def main():
    

    X = np.array([[2.], [5.], [3.], [10.]])
    Y = np.array([[8.], [25.], [9.], [40.]])

    theta = findTheta(X, Y)
    
    preds = predict(X, theta)
    
    plt.figure()
    plt.plot(X, Y,'b.')
    plt.plot(X, preds)
    plt.title("Question 1d")
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.show()
    
def findTheta(X,Y):
    m = X.shape[0]
    X = np.append(X, np.ones((m,1)), axis = 1)
    Y = Y.reshape(m,1)
    theta = np.dot(np.linalg.inv(np.dot(X.T,X)), np.dot(X.T,Y))
    return theta    

def predict(X, theta):
    X = np.append(X, np.ones((X.shape[0],1)), axis = 1)
    preds = np.dot(X, theta)
    
    return preds

if __name__ == "__main__":
    main()