# -*- coding: utf-8 -*-
"""
Created on Sun May  8 20:12:58 2022

@author: ynrob
"""

import numpy as np
import tensorflow as tf
from scipy.special import softmax

def main():
	K = [[10, 0, 0], [0, 10, 0], [0, 0, 10], [0, 0, 10]]
	V = [[1, 0], [10, 0], [100, 5], [1000, 6]]
	Q = [[0, 0, 10], [0, 10, 0], [10, 10, 0]]
	
	K = np.array(K)
	V = np.array(V)
	Q = np.array(Q)
	
	M = np.matmul(Q, K.T)
	attn = softmax(M)
	sum = 0
	for e in attn:
		sum += e
	print(sum)
	attn = np.matmul(attn, V)
	print(attn)
	sum = 0
	for e in attn:
		sum += e
	print(sum)

if __name__ == "__main__":
	main()