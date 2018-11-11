from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler
import pandas as pd 
import numpy as np 
import json


def create_file_json():
	url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

	#assign colum names to the dataset
	names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

	#read dataset to pandas dataframe
	df = pd.read_csv(url, names=names)
	X = df.ix[:,0:4].values
	y = df.ix[:,4].values

	X_std = StandardScaler().fit_transform(X)
	sklearn_pca = sklearnPCA(n_components=2)
	Y_learn = sklearn_pca.fit_transform(X_std)

	matrix_label = np.matrix(y).transpose()
	matrix_feature = np.matrix(Y_learn)

	matrix_general = np.concatenate((matrix_feature, matrix_label), axis=1)

	l = []
	for i in range(matrix_general.shape[0]-1):
		k=i*3
		d = {}
		d["x"] = matrix_general.item(k)
		d["y"] = matrix_general.item(k+1)
		d["label"] = matrix_general.item(k+2)
		l.append(d)

	with open('static/js/data2.json', 'w') as outfile:  
	    json.dump(l, outfile)