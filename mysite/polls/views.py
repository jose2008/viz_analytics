from django.shortcuts import render
from django.utils import timezone
import numpy as np
#from .models import Post
from polls.pca import create_file_json
import polls.somLib as somLib
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection  import cross_val_score
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_breast_cancer
import pandas as pd 
#import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch

#iris = load_breast_cancer()
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3) # 70% training and 30% test
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)



def most_common(lst, acurracy):
    if (lst[0]!=lst[1]) and (lst[0]!=lst[2]) and (lst[1]!=lst[2]):
        return lst[acurracy.index(max(acurracy))]
    else:
        return max(set(lst), key=lst.count)



def index(request):
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3) # 70% tra$
    acurracy_list = []
    sklearn_pca = sklearnPCA(n_components=2)


    

    ################################################# birch clustering #####################################################
    birch = Birch(branching_factor=50, n_clusters=3, threshold=0.5)
    birch.fit(iris.data)

    birch_std = StandardScaler().fit_transform(iris.data)
    birch_feature = sklearn_pca.fit_transform(birch_std)

    matrix__feature_birch = np.matrix(birch_feature)
    matrix_label_birch = np.matrix(birch.labels_).transpose()

    matrix_general_birch = np.concatenate((matrix__feature_birch, matrix_label_birch), axis=1)
    tolist_birch = matrix_general_birch.tolist()
    print(tolist_birch)
    v2 = birch.predict(iris.data)
    acurracy_list.append(metrics.adjusted_rand_score(iris.target, birch.predict(iris.data)) )
    #print(birch.predict(iris.data))
    print(tolist_birch)    

    #for i in range(len(iris.target)-1):
    #    #if tolist_knn[i][3] == 1:
    #    tolist_birch[i][2] = 1.0
    #    print("changeeeeee")

    #print(tolist_birch)


############################################### k-Means clustering ##################################################
    kmean = KMeans(n_clusters=3)
    kmean.fit(iris.data)

    kmean_std = StandardScaler().fit_transform(iris.data)
    kmean_feature = sklearn_pca.fit_transform(kmean_std)

    matrix_feature_kmean = np.matrix(kmean_feature)
    matrix_label_knn = np.matrix(kmean.labels_).transpose()
    index = np.matrix(np.arange(150)).transpose()

    matrix_general = np.concatenate((matrix_feature_kmean, matrix_label_knn), axis=1)
    matrix_test = np.concatenate((index, matrix_general),axis=1)
    tolist_knn = matrix_general.tolist()
    v1 = kmean.predict(iris.data)
    acurracy_list.append(metrics.adjusted_rand_score(iris.target, kmean.predict(iris.data)) )
    #print(kmean.predict(iris.data))


    from collections import Counter
    s = 0
    frequency = {}
    for i in range(len(iris.target)-1):
        if v1[i] == 0:
            count = frequency.get(v1[i],0)
            frequency[v2[i]] =count +1
    new_label =  max(frequency, key=frequency.get)


    #for i in range(len(iris.target)-1):
    #    if tolist_knn[i][3] == 2:
    #        aux =  tolist_knn[i][3]
    #        tolist_knn[i][3] = 1

    tolist_knn_final = tolist_knn


    #print(tolist_knn[1])
    #for i in range(len(iris.target)-1):
    #    #if tolist_knn[i][3] == 1:
    #    tolist_knn[i][3] = 1.0
    #    print("changeeeeee")



        



    ################################################## SOM clustering ######################################################
    df_train = pd.DataFrame( iris.data, columns=iris.feature_names)
    print(df_train.shape)
    df_test  = pd.DataFrame( X_test   , columns=iris.feature_names)
    df_original = df_train
    agri_som = somLib.SOM(1,3,4)
    df_train = df_train / df_train.max()
    df_test = df_test / df_test.max()
    agri_som.train(df_train.values,
              num_epochs=200,
              init_learning_rate=0.01
              )

    def predict(df):
        bmu, bmu_idx = agri_som.find_bmu(df.values)
        df['bmu'] = bmu
        df['bmu_idx'] =  bmu_idx[1]#bmu_idx
        return df
    clustered_df = df_train.apply(predict, axis=1)

    som_std = StandardScaler().fit_transform(iris.data)
    som_feature = sklearn_pca.fit_transform(som_std)

    matrix__feature_som = np.matrix(som_feature)
    matrix_label_som = np.matrix(clustered_df['bmu_idx'] ).transpose()
    print(matrix_label_som.shape)
    print(matrix__feature_som.shape)
    matrix_general_som = np.concatenate((matrix__feature_som, matrix_label_som), axis=1)
    tolist_som = matrix_general_som.tolist()
    acurracy_list.append(metrics.adjusted_rand_score(iris.target, clustered_df['bmu_idx']) )
    v3 = clustered_df['bmu_idx']


    ############################################## ensamble voting #############################################################
    voting = list()
    v = []
    for i in range(len(X_test)):
        voting.append( most_common( list([v1[i], v2[i], v3[i]]), acurracy_list) )

    l = np.array(voting)



    voting_std = StandardScaler().fit_transform(iris.data)
    voting_feature = sklearn_pca.fit_transform(voting_std)    

    matrix__feature_voting = np.matrix(voting_feature)
    matrix_label_voting = np.matrix(clustered_df['bmu_idx'] ).transpose()

    matrix_general_voting = np.concatenate((matrix__feature_voting, matrix_label_voting), axis=1)
    tolist_voting = matrix_general_voting.tolist()


    ############################################### seding model to view ###################################################
    list_model = []
    list_model.append(tolist_knn)
    list_model.append(tolist_birch)
    list_model.append(tolist_som)
    list_model.append(tolist_voting)
    print(tolist_som)
    return render(request,'index.html', {"model_list":list_model})