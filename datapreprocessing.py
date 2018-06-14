'''

Data preprocessing module for MR dataset.
Author: Harsh Tiku




This module will iterate through the entire document corpus
and encode each document into a 2400 length vector.

Giving data_size=(2000,2400) and label_size=(2000,1)


It will then make two directories, each for training data and test data
and split the data in 75-25 ratio


'''



from trainwordtovec import iterable
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np 
from sklearn.model_selection import train_test_split
import os

tokenizer=Tokenizer()


tokenizer.fit_on_texts(texts=iterable)


sequences=tokenizer.texts_to_sequences(texts=iterable)


vocab_size=len(tokenizer.word_index)


padded_sequences=pad_sequences(sequences=sequences,maxlen=2400)



labels=[0 for i in range(1000)]
labels.extend([1 for i in range(1000)])

padded_sequences=np.array(padded_sequences)
labels=np.array(labels)
labels=labels.flatten()
labels=labels.reshape(-1,1)

#print('data.shape = {}, Labels.shape={}'.format(padded_sequences.shape,labels.shape))


try:
	os.mkdir('./train')
	os.mkdir('./test')


	X_train,X_test,Y_train,Y_test=train_test_split(padded_sequences,labels)


	np.save(file='./train/train.npy',arr=X_train)
	np.save(file='./test/test.npy',arr=X_test)
	np.save(file='./train/trainlabels.npy',arr=Y_train)
	np.save(file='./test/testlabels.npy',arr=Y_test)


except FileExistsError as e:
	pass



#X_train,X_test,Y_train,Y_test=train_test_split(padded_sequences,labels)

'''
print('X_train.shape ={}'.format(X_train.shape))
print('X_test.shape={}'.format(X_test.shape))
print('Y_train.shape={}'.format(Y_train.shape))
print('Y_test.shape={}'.format(Y_test.shape))

print('train_test_split_ = 75-25')

'''

'''
np.save(file='./train/train.npy',arr=X_train)
np.save(file='./test/test.npy',arr=X_test)
np.save(file='./train/trainlabels.npy',arr=Y_train)
np.save(file='./test/testlabels.npy',arr=Y_test)

print('Datasets created in the home directory')

'''