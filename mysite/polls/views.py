from django.shortcuts import render
from django.utils import timezone
import numpy as np
from polls.pca import create_file_json
import polls.somLib as somLib
from sklearn.datasets import load_iris
from sklearn.model_selection  import cross_val_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd 
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch
import metis

#iris = load_breast_cancer()
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3) # 70% training and 30% test

sklearn_pca = sklearnPCA(n_components=2)
std = StandardScaler().fit_transform(iris.data)
feature = sklearn_pca.fit_transform(std)

def most_common(lst, acurracy):
    if (lst[0]!=lst[1]) and (lst[0]!=lst[2]) and (lst[1]!=lst[2]):
        return lst[acurracy.index(max(acurracy))]
    else:
        return max(set(lst), key=lst.count)


def index(request):
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3) # 70% tra$
    acurracy_list = []
    

    ############################################### k-Means clustering ##################################################
    kmean = KMeans(n_clusters=3)
    kmean.fit(iris.data)

    matrix_feature_kmean = np.matrix(feature)
    matrix_label_knn = np.matrix(kmean.labels_).transpose()
    index = np.matrix(np.arange(150)).transpose()

    matrix_general = np.concatenate((matrix_feature_kmean, matrix_label_knn), axis=1)
    tolist_knn = matrix_general.tolist()
    v1 = kmean.predict(iris.data)
    acurracy_list.append(metrics.adjusted_rand_score(iris.target, kmean.predict(iris.data)) )


    ################################################# birch clustering #####################################################
    birch = Birch(branching_factor=50, n_clusters=3, threshold=0.5)
    birch.fit(iris.data)

    matrix__feature_birch = np.matrix(feature)
    matrix_label_birch = np.matrix(birch.labels_).transpose()

    matrix_general_birch = np.concatenate((matrix__feature_birch, matrix_label_birch), axis=1)
    tolist_birch = matrix_general_birch.tolist()
    v2 = birch.predict(iris.data)
    acurracy_list.append(metrics.adjusted_rand_score(iris.target, birch.predict(iris.data)) )

    tolist_birch_test = tolist_birch
    frequency = {}
    for i in range(len(iris.target)):
        if v2[i] == 0:
            count = frequency.get(v1[i], 0)
            frequency[v1[i]] = count +1
    value_0 = max(frequency, key=frequency.get)

    frequency2 = {}
    for i in range(len(iris.target)):
        if v2[i] == 1:
            count = frequency2.get(v1[i], 0)
            frequency2[v1[i]] = count +1
    value_1 = max(frequency2, key=frequency2.get)

    frequency3 = {}
    for i in range(len(iris.target)):
        if v2[i] == 2:
            count = frequency3.get(v1[i], 0)
            frequency3[v1[i]] = count +1
    value_2 = max(frequency3, key=frequency3.get)

    list_dict = list(frequency.keys())
    if(list_dict[0]!=value_1 and list_dict[0]!=value_2):
        value_0 = list_dict[0]
    else:
        value_0 = list_dict[1]


    for i in range(len(iris.target)):
        if v2[i] == 1:
            tolist_birch_test[i][2] = value_1

    for i in range(len(iris.target)):
        if v2[i] == 2:
            tolist_birch_test[i][2] = value_2

    for i in range(len(iris.target)):
        if v2[i] == 0:
            tolist_birch_test[i][2] = value_0

    for i in range(len(iris.target)):
        v2[i] = tolist_birch[i][2]


    ################################################## SOM clustering ######################################################
    df_train = pd.DataFrame( iris.data, columns=iris.feature_names)
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

    matrix__feature_som = np.matrix(feature)
    matrix_label_som = np.matrix(clustered_df['bmu_idx'] ).transpose()
    matrix_general_som = np.concatenate((matrix__feature_som, matrix_label_som), axis=1)
    tolist_som = matrix_general_som.tolist()
    acurracy_list.append(metrics.adjusted_rand_score(iris.target, clustered_df['bmu_idx']) )
    v3 = clustered_df['bmu_idx']


    tolist_som_test = tolist_som
    frequency_som_1 = {}
    for i in range(len(iris.target)):
            if v3[i] == 0:
                count = frequency_som_1.get(v1[i], 0)
                frequency_som_1[v1[i]] = count +1
    value_som_0 = max(frequency_som_1, key=frequency_som_1.get)


    frequency_som_2 = {}
    for i in range(len(iris.target)):
            if v3[i] == 1:
                count = frequency_som_2.get(v1[i], 0)
                frequency_som_2[v1[i]] = count +1
    value_som_1 = max(frequency_som_2, key=frequency_som_2.get)


    frequency_som_3 = {}
    for i in range(len(iris.target)):
            if v3[i] == 2:
                count = frequency_som_3.get(v1[i], 0)
                frequency_som_3[v1[i]] = count +1
    value_som_2 = max(frequency_som_3, key=frequency_som_3.get)


    list_dict2 = list(frequency_som_1.keys())
    if(list_dict2[0]!=value_som_1 and list_dict2[0]!=value_som_2):
        value_som_0= list_dict2[0]
    else:
        value_som_0 = list_dict2[1]

    
    for i in range(len(iris.target)):
        if v3[i] == 1:
            tolist_som_test[i][2] = value_som_1

    for i in range(len(iris.target)):
        if v3[i] == 2:
            tolist_som_test[i][2] = value_som_2


    for i in range(len(iris.target)):
        if v3[i] == 0:
            tolist_som_test[i][2] = value_som_0


    for i in range(len(iris.target)):
        v3[i] = tolist_som_test[i][2]


    ############################################## ensamble voting #############################################################
    voting = list()
    for i in range(len(iris.data)):
        voting.append( most_common( list([v1[i], v2[i], v3[i]]), acurracy_list) )

    l = np.array(voting)

    matrix__feature_voting = np.matrix(feature)
    matrix_label_voting = np.matrix(l ).transpose()
    matrix_general_voting = np.concatenate((matrix__feature_voting, matrix_label_voting), axis=1)
    tolist_voting = matrix_general_voting.tolist()


    ############################################ make matriz k-means && birch && som ##########################################################
    s1_knn = set()
    s2_knn = set()
    s3_knn = set()
    s1_birch = set()
    s2_birch = set()
    s3_birch = set()
    s1_som = set()
    s2_som = set()
    s3_som = set()

    list_set = []
    list_set.append(s1_knn)
    list_set.append(s2_knn)
    list_set.append(s3_knn)
    list_set.append(s1_birch)
    list_set.append(s2_birch)
    list_set.append(s3_birch)
    list_set.append(s1_som)
    list_set.append(s2_som)
    list_set.append(s2_som)


    n = len(kmean.labels_)
    for i in range(n):
        if kmean.labels_[i] == 0:
            s1_knn.add(i)
        if kmean.labels_[i] == 1:
            s2_knn.add(i)
        if kmean.labels_[i] == 2:
            s3_knn.add(i)
        if birch.labels_[i] == 0:
            s1_birch.add(i)
        if birch.labels_[i] == 1:
            s2_birch.add(i)
        if birch.labels_[i] == 2:
            s3_birch.add(i)
        if clustered_df['bmu_idx'][i] == 0:
            s1_som.add(i)
        if clustered_df['bmu_idx'][i] == 1:
            s2_som.add(i)
        if clustered_df['bmu_idx'][i] == 2:
            s3_som.add(i)
    
    matrix = np.zeros((n, n))
    matrix_birch = np.zeros((n, n))
    matrix_som = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if (i in s1_knn and j in s1_knn) or (i in s2_knn and j in s2_knn) or (i in s3_knn and j in s3_knn):
                matrix[i][j] = 1
            else:
                matrix[i][j] = 0
            if (i in s1_birch and j in s1_birch) or (i in s2_birch and j in s2_birch) or (i in s3_birch and j in s3_birch):
                matrix_birch[i][j] = 1
            else:
                matrix_birch[i][j] = 0
            if (i in s1_som and j in s1_som) or (i in s2_som and j in s2_som) or (i in s3_som and j in s3_som):
                matrix_som[i][j] = 1
            else:
                matrix_som[i][j] = 0

    h = np.zeros((n, len(list_set)))
    list_matriz_kmean = matrix.tolist()
    list_matriz_birch = matrix_birch.tolist()
    list_matriz_som = matrix_som.tolist()


    ############################################### matrix ghpa ############################################################

    for i in range(n):
        for j in range(len(list_set)):
            if i in list_set[j]:
                h[i][j] = 1
            else:
                h[i][j] = 0
    h_t = h.transpose()
    s = (0.3)*h.dot(h_t)
    list_matrix_s = s.tolist()

    ############################################ algorithm ghpa  ###########################################################
    list_ad = []
    for i in range(n):
        list_tmp = []
        for j in range(n):
            if(list_matrix_s[i][j] == 0):
                continue
            else:
                t = (j, int(round(10*list_matrix_s[i][j])))
                list_tmp.append(t)
        list_ad.append(list_tmp)

    cuts, parts = metis.part_graph(list_ad, 3, recursive = False, dbglvl=metis.METIS_DBG_ALL)

    matrix__feature_cspa = np.matrix(feature)

    matrix_label_cspa = np.matrix(parts).transpose()
    matrix_general_cspa = np.concatenate((matrix__feature_cspa, matrix_label_cspa), axis=1)
    tolist_cspa = matrix_general_cspa.tolist()

    v4 = parts
    tolist_cspa_test = tolist_cspa
    frequency_cspa = {}
    for i in range(len(iris.target)):
        if v4[i] == 0:
            count = frequency_cspa.get(v1[i], 0)
            frequency_cspa[v1[i]] = count +1
    val_0 = max(frequency_cspa, key=frequency_cspa.get)

    frequency2_cspa = {}
    for i in range(len(iris.target)):
        if v4[i] == 1:
            count = frequency2_cspa.get(v1[i], 0)
            frequency2_cspa[v1[i]] = count +1
    val_1 = max(frequency2_cspa, key=frequency2_cspa.get)

    frequency3_cspa = {}
    for i in range(len(iris.target)):
        if v4[i] == 2:
            count = frequency3_cspa.get(v1[i], 0)
            frequency3_cspa[v1[i]] = count +1
    val_2 = max(frequency3_cspa, key=frequency3_cspa.get)

    list_dict_cspa = list(frequency_cspa.keys())
    if(list_dict_cspa[0]!=val_1 and list_dict_cspa[0]!=val_2):
        val_0 = list_dict_cspa[0]
    else:
        val_0 = list_dict_cspa[1]


    for i in range(len(iris.target)):
        if v4[i] == 1:
            tolist_cspa_test[i][2] = val_1

    for i in range(len(iris.target)):
        if v4[i] == 2:
            tolist_cspa_test[i][2] = val_2

    for i in range(len(iris.target)):
        if v4[i] == 0:
            tolist_cspa_test[i][2] = val_0

    for i in range(len(iris.target)):
        v4[i] = tolist_cspa[i][2]









    ############################################### seding model to view ###################################################
    list_model = []
    list_model.append(tolist_knn)
    list_model.append(tolist_birch_test)
    list_model.append(tolist_som_test)
    list_model.append(tolist_voting)
    list_model.append(list_matriz_kmean)
    list_model.append(list_matriz_birch)
    list_model.append(list_matriz_som)
    list_model.append(list_matrix_s)
    list_model.append(tolist_cspa_test)
    return render(request,'index.html', {"model_list":list_model})