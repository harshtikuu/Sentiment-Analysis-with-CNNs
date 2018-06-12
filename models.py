from keras.layers import Embedding,Reshape,Dropout,Dense,Conv2D,Flatten,Lambda,Input,Concatenate
from keras.models import Model 
from gensim.models import KeyedVectors
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from trainwordtovec import iterable
import numpy as np 


vocab_size=43297
embedding_dim=300
input_length=2400
n_filters=100



def get_random_model(print_summary=False):

	inp=Input(shape=(input_length,),name='StaticWord2VecInput')
	embed=Embedding(input_dim=vocab_size,output_dim=embedding_dim,input_length=input_length)(inp)
	embed=Lambda(lambda x: K.expand_dims(x))(embed)
	x1=Conv2D(filters=n_filters,kernel_size=(3,embedding_dim),strides=1,padding='valid',activation='tanh',name='Convolution_s3')(embed)
	r1=Reshape((-1,n_filters))(x1)
	maxpool1=Lambda(function=lambda x: K.max(x,axis=1),name='maxpool1')(r1)




	x2=Conv2D(filters=n_filters,kernel_size=(4,embedding_dim),strides=1,padding='valid',activation='tanh',name='Convolution_s4')(embed)
	r2=Reshape((-1,n_filters))(x2)
	maxpool2=Lambda(function=lambda x: K.max(x,axis=1),name='maxpool2')(r2)


	x3=Conv2D(filters=n_filters,kernel_size=(5,embedding_dim),strides=1,padding='valid',activation='tanh',name='Convolution_s5')(embed)
	r3=Reshape((-1,n_filters))(x3)
	maxpool3=Lambda(function=lambda x: K.max(x,axis=1),name='maxpool3')(r3)



	concatenated=Concatenate(axis=1,name='concatenated')([maxpool1,maxpool2,maxpool3])
	concatenated=Dropout(0.5)(concatenated)

	dense=Dense(1,activation='sigmoid',name='outputlayers')(concatenated)

	model=Model(inputs=[inp],outputs=[dense])

	model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

	if print_summary:
		model.summary()

	return model 


def get_keras_embedding(modelpath,finetune=False): 

	global vocab_size,embedding_dim

	t=Tokenizer()
	t.fit_on_texts(iterable)

	trained_word_model=KeyedVectors.load_word2vec_format(modelpath)

	embedding_matrix=np.zeros((vocab_size,embedding_dim))

	for word,i in t.word_index.items():
		try:
			embedding_vector=trained_word_model[word]
		except:
			embedding_vector=np.random.random((embedding_dim,))

		embedding_matrix[i]=embedding_vector

	layer=Embedding(input_dim=vocab_size,output_dim=embedding_dim,weights=[embedding_matrix],input_length=input_length,trainable=finetune)

	return layer






def get_pretrained_model(modelpath='./model.bin',finetune=False,print_summary=False):
	
	embedlayer=get_keras_embedding(modelpath,finetune)



	inp=Input(shape=(input_length,),name='StaticWord2VecInput')
	embed=embedlayer(inp)
	embed=Lambda(lambda x: K.expand_dims(x))(embed)
	x1=Conv2D(filters=n_filters,kernel_size=(3,embedding_dim),strides=1,padding='valid',activation='tanh',name='Convolution_s3')(embed)
	r1=Reshape((-1,n_filters))(x1)
	maxpool1=Lambda(function=lambda x: K.max(x,axis=1),name='maxpool1')(r1)




	x2=Conv2D(filters=n_filters,kernel_size=(4,embedding_dim),strides=1,padding='valid',activation='tanh',name='Convolution_s4')(embed)
	r2=Reshape((-1,n_filters))(x2)
	maxpool2=Lambda(function=lambda x: K.max(x,axis=1),name='maxpool2')(r2)


	x3=Conv2D(filters=n_filters,kernel_size=(5,embedding_dim),strides=1,padding='valid',activation='tanh',name='Convolution_s5')(embed)
	r3=Reshape((-1,n_filters))(x3)
	maxpool3=Lambda(function=lambda x: K.max(x,axis=1),name='maxpool3')(r3)



	concatenated=Concatenate(axis=1,name='concatenated')([maxpool1,maxpool2,maxpool3])
	concatenated=Dropout(0.5)(concatenated)

	dense=Dense(1,activation='sigmoid',name='outputlayers')(concatenated)

	model=Model(inputs=[inp],outputs=[dense])

	model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

	if print_summary:
		model.summary()

	return model 


def get_multichannel_model(modelpath='./model.bin',print_summary=False):

	embedlayer1=get_keras_embedding(modelpath='./model.bin',finetune=True)
	embedlayer2=get_keras_embedding(modelpath='./model.bin',finetune=False)


	inp=Input(shape=(input_length,))
	embed1=embedlayer1(inp)
	embed2=embedlayer2(inp)

	embed1=Lambda(lambda x: K.expand_dims(x,axis=-1))(embed1)
	embed2=Lambda(lambda x: K.expand_dims(x,axis=-1))(embed2)




	embed=Concatenate()([embed1,embed2])

	x1=Conv2D(filters=n_filters,kernel_size=(3,embedding_dim),strides=1,padding='valid',activation='tanh',name='Convolution_s3')(embed)
	r1=Reshape((-1,n_filters))(x1)
	maxpool1=Lambda(function=lambda x: K.max(x,axis=1),name='maxpool1')(r1)




	x2=Conv2D(filters=n_filters,kernel_size=(4,embedding_dim),strides=1,padding='valid',activation='tanh',name='Convolution_s4')(embed)
	r2=Reshape((-1,n_filters))(x2)
	maxpool2=Lambda(function=lambda x: K.max(x,axis=1),name='maxpool2')(r2)


	x3=Conv2D(filters=n_filters,kernel_size=(5,embedding_dim),strides=1,padding='valid',activation='tanh',name='Convolution_s5')(embed)
	r3=Reshape((-1,n_filters))(x3)
	maxpool3=Lambda(function=lambda x: K.max(x,axis=1),name='maxpool3')(r3)



	concatenated=Concatenate(axis=1,name='concatenated')([maxpool1,maxpool2,maxpool3])
	concatenated=Dropout(0.5)(concatenated)

	dense=Dense(1,activation='sigmoid',name='outputlayers')(concatenated)

	model=Model(inputs=[inp],outputs=[dense])

	model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

	if print_summary:
		model.summary()


	return model 






