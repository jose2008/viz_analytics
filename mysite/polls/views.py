from django.shortcuts import render
from django.utils import timezone
import json
import numpy as np
# Create your views here.
#from .models import Post
from polls.pca import create_file_json
#make algorithm of classification
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection  import cross_val_score
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_breast_cancer


#import sys
#sys.path.append('../../')


create_file_json()

iris = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3) # 70% training and 30% test
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)




def index(request):
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3) # 70% tra$
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    mydict = dict(np.ndenumerate(cm))

    for key in mydict.keys():
        if type(key) is not str:
            try:
                mydict[str(key)] = mydict[key]
            except:
                try:
                    mydict[repr(key)] = mydict[key]
                except:
                    pass
            del mydict[key]



    list = [1,2,3,4]
    x = np.matrix(cm)
    y2 = x.tolist()





    #generation of matrix of kmeans
    import pandas as pd 
    #import numpy as np
    from sklearn.cluster import KMeans
    from sklearn import datasets
    from sklearn.decomposition import PCA as sklearnPCA
    from sklearn.preprocessing import StandardScaler
    import json
    from sklearn.cluster import Birch


    #iris = datasets.load_iris()
    x = iris.data
    kmean = KMeans(n_clusters=3)
    y = kmean.fit(x)


    #print(x)
    #print(kmean.labels_)
    #print("exit....")



    x_std = StandardScaler().fit_transform(x)
    sklearn_pca = sklearnPCA(n_components=2)
    feature = sklearn_pca.fit_transform(x_std)


    matrix_x = np.matrix(feature)
    matrix_y = np.matrix(kmean.labels_).transpose()


    #print(matrix_x)
    #print(matrix_y)

    matrix_general = np.concatenate((matrix_x, matrix_y), axis=1)
    #print(matrix_general)


    y3 = matrix_general.tolist()





    x2 = iris.data
    kmean2 = Birch(branching_factor=50, n_clusters=3, threshold=0.5)
    y4 = kmean2.fit(x2)


    #print(x)
    #print(kmean.labels_)
    #print("exit....")



    x_std2 = StandardScaler().fit_transform(x2)
    sklearn_pca2 = sklearnPCA(n_components=2)
    feature3 = sklearn_pca2.fit_transform(x_std2)


    matrix_x2 = np.matrix(feature3)
    matrix_y2 = np.matrix(kmean2.labels_).transpose()


    #print(matrix_x)
    #print(matrix_y)

    matrix_general2 = np.concatenate((matrix_x2, matrix_y2), axis=1)
    y4 = matrix_general2.tolist()






    list_model = []
    list_model.append(y3)
    list_model.append(y4)







    l = []
    for i in range(matrix_general.shape[0]-1):
        k = i*3
        d = {}
        d["x"] = matrix_general.item(k)
        d["y"] = matrix_general.item(k+1)
        d["label"] = matrix_general.item(k+2)
        l.append(d)

    with open('data3.json','w') as outfile:
        json.dump(l, outfile)




    #print("---------------------->"+y)

    days = 2
    count_data = 5
    data = {"label":days, "days_of_data":count_data}
    my_data = {'my_data':json.dumps(data)}
    #posts = Post.objects.filter(published_date__lte=timezone.now()).order_by('published_date')
    return render(request,'index.html', {"m2":list_model})
