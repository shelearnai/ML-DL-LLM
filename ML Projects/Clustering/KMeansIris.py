import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import numpy as np

#Read input
df=pd.read_csv('Iris1.csv')

#Change class label Species to numeric number
y=pd.factorize(df['Species'])[0].astype(np.int64)
df.drop(['Id','Species'],1,inplace=True)
df.fillna(0,inplace=True)

X=np.array(df)
y=np.array(y)

#Clustering
clf=KMeans(n_clusters=2)
clf.fit(X)

correct=0
for i in range(len(X)):
	pred=np.array(X[i].astype(float))
	pred=pred.reshape(-1,len(pred))
	predict_result=clf.predict(pred)
	if predict_result[0]==y[i]:
		correct+=1

print("correct count : {}".format(correct))
print("correct percent : {}".format(correct/len(X)))
