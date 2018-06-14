# Code for the paper [here](http://www.gitxiv.com/posts/4KgE7qCM8ZmPHiju4/convolutional-neural-networks-for-sentence-classification).

### The models have been implemented in Keras.

### The Google Word2vec embeddings file can be downloaded from [here](https://code.google.com/archive/p/word2vec/) for loading into the non-random versions of the models.


### The dataset is present in the source directory itself.

### The paper is also present in the source directory


# Steps:


#### Preprocessing raw text and splitting into train-test

```bash
$ python3 dataprocessing.py
```

```python



#Loading CNN_rand
from models import get_random_model
import numpy as np

model=get_random_model()
X_train=np.load(trainfile)
Y_train=np.load(trainlabelsfile)
model.fit(X_train,Y_train)

#Loading CNN_static
from models import get_pretrained_model
import numpy as np

model=get_pretrained_model(word2vecmodelpath,finetune=False)
X_train=np.load(trainfile)
Y_train=np.load(trainlabelsfile)
model.fit(X_train,Y_train)

#Loading CNN_non_static
from models import get_pretrained_model
import numpy as np

model=get_pretrained_model(word2vecmodelpath,finetune=True)
X_train=np.load(trainfile)
Y_train=np.load(trainlabelsfile)
model.fit(X_train,Y_train)

#Loading CNN_multi_channel

from models import get_multichannel_model
import numpy as np

model=get_multichannel_model(word2vecmodelpath)
X_train=np.load(trainfile)
Y_train=np.load(trainlabelsfile)
model.fit(X_train,Y_train)
```
##### You can simulate the entire process by changing some parameters in code and runnning
```bash
$ python3 traintest.py
```
##### Since the Google Word2vec file is too large (1.5GB) I trained a myown word2vec model on a shorter text corpus. The Google model can however be downloaded and used as pretrained weights into the embedding layer.


#####  All the hyperparameters mentioned in the paper have been incorporated into the models as such but instead of adadelta optimiser, adam is used. Also instead of using softmax on the penultimate layer,I used sigmoid and tried experimenting with kernel constraints of 2,4,8 but got better values if I just used no kernel constraints.

##### I trained the random model on a kaggle kernel and achieved 80% accuracy on test data after training for like 10 epochs on training data. This is better than 76% accuracy given in the paper.

## The link to the kaggle kernel is [here](https://www.kaggle.com/harshtiku/convnets-for-sentence-classification)

## The github link to the code is [here](https://github.com/blackeagle01/Sentiment-Analysis-with-CNNs)


