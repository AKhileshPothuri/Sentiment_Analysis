import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

df = pd.read_csv('train.csv')
print(list(df['text'])[0])

x = df.iloc[:,1].values
y = df.iloc[:,0].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
categories = [0,1,2]
print('Training class distributions summary: {}'.format(Counter(y_train)))
print('Test class distributions summary: {}'.format(Counter(y_test)))
