import pandas as pd
from textblob import TextBlob
df= pd.read_csv("test_data.csv")
for i in range(len(df['text'])):
    print(TextBlob(df.ix[i,'text'].sentiment))
