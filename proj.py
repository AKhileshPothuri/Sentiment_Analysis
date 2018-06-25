import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer


df = pd.read_csv("train.csv")
nltk.download('stopwords')

i = nltk.corpus.stopwords.words('english')

stopwords = set(i)

stopwords.remove("not")
stopwords.remove("against")
stopwords.remove("no")

def preprocess(x):
    
    if(type(x)==str):
        x = re.sub('[^a-z\s]', '', x.lower())
        x = re.sub(r'[^\w\s]', "", x)
        x = [w for w in x.split() if w not in set(stopwords)]
        return x     




g=[]
for i in range(0,len(df)):
    y = preprocess(df['text'][i])
    g.append(y)

count = 0
for i in g:
    t = " ".join(i)
    df.ix[count,'text']=t
    count+=1

df.to_csv('preprocessed.csv')


#print(df.head(10))

#df.to_csv("preprocessed_data.csv", encoding='utf-8', index=False)

#for i in range(0,len(df)):
 #   print(g[i])
    
