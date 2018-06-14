import datapreprocessing
from models import get_random_model,get_pretrained_model,get_multichannel_model
import matplotlib.pyplot as plt
import numpy as np



def train(modeltype='random',epochs=7,word_embeddings='./samplemodel.bin'):

	if  modeltype not in ['random','static','finetune','multi_channel']:
		raise ValueError('Invalid Argument')


	X_train=np.load('./train/train.npy')
	Y_train=np.load('./train/trainlabels.npy')

	if modeltype=='random':
		model=get_random_model()
		model.summary()
		history=model.fit(X_train,Y_train,validation_split=0.1,epochs=epochs)

	elif modeltype=='static':
		model=get_pretrained_model(finetune=False,modelpath=word_embeddings)
		model.summary()
		history=model.fit(X_train,Y_train,validation_split=0.1,epochs=epochs)

	elif modeltype=='finetune':
		model=get_pretrained_model(finetune=True,modelpath=word_embeddings)
		model.summary()
		history=model.fit(X_train,Y_train,validation_split=0.1,epochs=epochs)

	elif modeltype=='multi_channel':
		model=get_pretrained_model(finetune=True,modelpath=word_embeddings)
		model.summary()
		history=model.fit(X_train,Y_train,validation_split=0.1,epochs=epochs)

	else:
		pass


	return model,history


def generate_plot(history):
	plt.plot(history.history['loss'])
	plt.xlabel('Epochs')
	plt.ylabel('Loss')

	plt.show()


def test(model):
	X_test=np.load('./test/test.npy')
	Y_test=np.load('./test/testlabels.npy')

	loss,accuracy=model.evaluate(X_test,Y_test)

	return loss,accuracy

if __name__=='__main__':

	history,model=train()

	print('Model trained')


	generate_plot(history)


	loss,accuracy=test(model)

	print('Accuracy on test_data ={}'.format(accuracy))
	print('Loss on test_data ={}'.format(accuracy))












