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

print(padded_sequences.shape,labels.shape)



os.mkdir('./train')
os.mkdir('./test')


X_train,X_test,Y_train,Y_test=train_test_split(padded_sequences,labels)

np.save(file='./train/train.npy',arr=X_train)
np.save(file='./test/test.npy',arr=X_test)
np.save(file='./train/trainlabels.npy',arr=Y_train)
np.save(file='./test/testlabels.npy',arr=Y_test)

print('Datasets created in the home directory')

