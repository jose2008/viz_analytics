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
    y = x.tolist()


    #print("---------------------->"+y)

    days = 2
    count_data = 5
    data = {"label":days, "days_of_data":count_data}
    my_data = {'my_data':json.dumps(data)}
    #posts = Post.objects.filter(published_date__lte=timezone.now()).order_by('published_date')
    return render(request,'index.html', {"m2":y})
