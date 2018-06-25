from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("LIWC2015 Results_removed_comments.csv")
X = df.iloc[:,0:93].values
Y = df.iloc[:,93].values
from sklearn.cross_validation import train_test_split

pca = PCA(n_components=20)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents)
finalDf = pd.concat([principalDf, df[['A']]], axis = 1)
#print(pd.DataFrame(pca.components_))
print(finalDf)
finalDf.to_csv('PCA_extracted_features.csv')
