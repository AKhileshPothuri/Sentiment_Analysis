from __future__ import print_function
import numpy as np
import pandas as pd
np.random.seed(1337)  # for reproducibility

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, GRU
from sklearn.model_selection import train_test_split


batch_size = 5000

df = pd.read_csv('train.csv')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['text'])

sequences = tokenizer.texts_to_sequences(df['text'])
l = []
for li in sequences:
    l.append(max(li)-min(li))

input_size = max(l)+1
    
X = pad_sequences(sequences,maxlen= 750)
Y = to_categorical(df['label'])
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

 

model_lstm  = Sequential()
model_lstm.add(Embedding(input_size, 256, input_length=750))
model_lstm.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))
model_lstm.add(Dense(3 , activation='relu'))
model_lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model_lstm.fit(x_train,y_train,batch_size=batch_size,nb_epoch=10)

score, acc = model.evaluate(x_test,y_test)
print("Score: ",score)
print("Accuracy: ",acc)
