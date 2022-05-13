# -*- coding: utf-8 -*-
"""
Created on Sun May  8 19:47:28 2022

@author: ynrob
"""

import numpy as np
import tensorflow as tf

def main():
	tf.compat.v1.disable_eager_execution()
	corpus_raw = "Richard Nixon is gone now, and I am poorer for it. He was the real thing -- a political monster straight out of Grendel and a very dangerous enemy. He could shake your hand and stab you in the back at the same time. He lied to his friends and betrayed the trust of his family. Not even Gerald Ford, the unhappy ex-president who pardoned Nixon and kept him out of prison, was immune to the evil fallout. Ford, who believes strongly in Heaven and Hell, has told more than one of his celebrity golf partners that \"I know I will go to hell, because I pardoned Richard Nixon.\""
	# convert to lower case
	corpus_raw = corpus_raw.lower()
	corpus_raw = corpus_raw.replace(",", "")
	words = []
	for word in corpus_raw.split():
	    if word != '.': # because we don't want to treat . as a word
	        words.append(word.replace(".", ""))
	words = set(words) # so that all duplicate words are removed
	   
	global word2int
	word2int = {}
	int2word = {}
	vocab_size = len(words) # gives the total number of unique words
	print('Vocab size: ', vocab_size)
	print('Words:', words)
	
	for i,word in enumerate(words):
	    word2int[word] = i
	    int2word[i] = word
		
	raw_sentences = corpus_raw.split('.')
	sentences = []
	print( 'Total number of sentenses are:', len(raw_sentences))
	for i, sentence in enumerate(raw_sentences):
		sentences.append(sentence.split())
	
	#using fixed window size, pairs of word are created for input
	WINDOW_SIZE = 2
	global data
	data = []
	for sentence in sentences:
	    for word_index, word in enumerate(sentence):
	        for nb_word in sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1] : 
	            if nb_word != word:
	                data.append([word, nb_word])
	print('\n Original sentence:: ', corpus_raw)
	print('\n ************************Input is getting prepared ***********************')  
	for i, eachpair in enumerate(data):
	    print(i, ':', eachpair)
		
	# function to convert numbers to one hot vectors
	def to_one_hot(data_point_index, vocab_size):
	    temp = np.zeros(vocab_size)
	    temp[data_point_index] = 1
	    return temp
	
	x_train = [] # input word
	y_train = [] # output word
	labelss = []
	for i, data_word in enumerate(data):
	    
	    x_train.append((word2int[ data_word[0] ]))
	    if data_word[0] not in labelss:
	        labelss.append(data_word[0] )
	    y_train.append((word2int[ data_word[1] ]))
	
	# convert them to numpy arrays
	x_train = np.asarray(x_train)
	y_train = np.asarray(y_train)
	print('words: ', words)
	print('X : \n', x_train)
	print('\nY: \n', y_train)
		
	
	y_train = y_train.reshape((-1, 1))
	
	BATCH_SIZE = 10
	VOCAB_SIZE = vocab_size #12
	EMBED_SIZE = 5
	NUM_SAMPLED= 6
	LEARNING_RATE =1.0 # 1e-1
	X = tf.compat.v1.placeholder(dtype=tf.int32, shape=[BATCH_SIZE])
	Y = tf.compat.v1.placeholder(dtype=tf.int32, shape=[BATCH_SIZE, 1])
	
	with tf.device("/cpu:0"):
	    embed_matrix = tf.Variable(tf.random.uniform([VOCAB_SIZE, EMBED_SIZE], -1.0, 1.0)) #12,5 
	    embed = tf.nn.embedding_lookup(embed_matrix, X) #50 , 3
		
	nce_weight = tf.Variable(tf.random.uniform([VOCAB_SIZE, EMBED_SIZE],-1.0,1.0)) # (12, 5)
	nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]))#12
	
	loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                     biases=nce_bias,
                                     labels=Y,
                                     inputs=embed,
                                     num_sampled=NUM_SAMPLED,
                                     num_classes=VOCAB_SIZE
                                    ))
	optimizer = tf.compat.v1.train.AdamOptimizer(1e-1).minimize(loss)
	epochs = 10000
	with tf.compat.v1.Session() as sess:
	    sess.run(tf.compat.v1.global_variables_initializer())
	    
	    for epoch in range (epochs):
	        batch_inputs, batch_labels = get_batch(BATCH_SIZE)
	        _,loss_val = sess.run([optimizer,loss],feed_dict={X: batch_inputs, Y: batch_labels})
	    
	        if epoch % 1000 == 0:
	            print("Loss at", epoch, loss_val )
	        
	    temp = embed_matrix.eval()
	
	from sklearn.decomposition import PCA
	pca = PCA(n_components=2)
	trained_embeedings = pca.fit_transform(temp)
		
	import matplotlib.pyplot as plt
	#show word2vec if dim is 2
	if trained_embeedings.shape[1] == 2:
	    #labels = data[:10] #Show top 10 words
	#     plt.xlim(-2.5, 2.4)
	#     plt.ylim(-2.0, 2.2)
	    for i, label in enumerate(labelss):
	        x,y = trained_embeedings[i,:]
	        plt.scatter(x,y)
	        plt.annotate(label, xy=(x,y), xytext=(9,3),textcoords='offset points', ha='right', va='bottom')
	        #plt.savefig('word2vev.png')
	    plt.show()
	
	
def get_batch(size):
    assert size<len(data)
    X=[]
    Y=[]
    rdm = np.random.choice(range(len(data)), size, replace=False)
    
    for r in rdm:
        X.append(word2int[data[r][0]])
        Y.append([word2int[data[r][1]]])
    return X, Y
	
if __name__ == "__main__":
	main()