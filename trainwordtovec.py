import os
from keras.preprocessing.text import Tokenizer,text_to_word_sequence 
from gensim.models import Word2Vec

negfilepath='./review_polarity/txt_sentoken/neg/'
posfilepath='./review_polarity/txt_sentoken/pos/'



filenames=[]
labels=[0 for i in range(1000)]
labels.extend([1 for i in range(1000)])


for i in os.listdir(negfilepath):
	filenames.append(os.path.join(negfilepath,i))



for i in os.listdir(posfilepath):
	filenames.append(os.path.join(posfilepath,i))


texts=[]

for file in filenames:
	string=open(file).read()
	texts.append(string)

def generator_function():
    global texts
    for i in texts:
        sequence=text_to_word_sequence(i)
        
        yield sequence


class MakeIter(object):
    def __init__(self, generator_func, **kwargs):
        self.generator_func = generator_func
        self.kwargs = kwargs
    def __iter__(self):
        return self.generator_func(**self.kwargs)


iterable=MakeIter(generator_func=generator_function)

if __name__ == '__main__':


		print('Training word Embeddings:')

		model=Word2Vec(sentences=iterable,size=300,workers=6,min_count=1)

		print('Training Complete')
		path='./model.bin'


		model.wv.save_word2vec_format(fname=path)

		print("Embedding model saved at {}".format(path))







