import tensorflow as tf
from keras import backend as K
import numpy as np
from keras.layers import Dense,Reshape,Flatten,Lambda,Conv2D,Input,Dropout,Concatenate
from keras.models import Model,Sequential

def StaticConvModel():

    inp=Input(shape=(100,300,1),name='StaticWord2VecInput')
    x1=Conv2D(filters=100,kernel_size=(3,300),strides=1,padding='valid',activation='tanh',name='Convolution_s3')(inp)
    r1=Reshape((98,100))(x1)
    maxpool1=Lambda(function=lambda x: K.max(x,axis=1),name='maxpool1')(r1)




    x2=Conv2D(filters=100,kernel_size=(4,300),strides=1,padding='valid',activation='tanh',name='Convolution_s4')(inp)
    r2=Reshape((97,100))(x2)
    maxpool2=Lambda(function=lambda x: K.max(x,axis=1),name='maxpool2')(r2)


    x3=Conv2D(filters=100,kernel_size=(5,300),strides=1,padding='valid',activation='tanh',name='Convolution_s5')(inp)
    r3=Reshape((96,100))(x3)
    maxpool3=Lambda(function=lambda x: K.max(x,axis=1),name='maxpool3')(r3)



    concatenated=Concatenate(axis=1,name='concatenated')([maxpool1,maxpool2,maxpool3])
    concatenated=Dropout(0.5)(concatenated)

    dense=Dense(10,activation='softmax',name='outputlayers')(concatenated)

    model=Model(inputs=[inp],outputs=[dense])

    model.summary()
    return model



if __name__ =='__main__':
	model=StaticConvModel()