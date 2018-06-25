import pandas as pd
import textblob
df= pd.read_csv("test_data.csv")
for i in range(len(df['text'])):
    print(TextBlob(df.ix[i,'text'].sentiment))
