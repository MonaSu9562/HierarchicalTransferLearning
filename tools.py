import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
import glob
import os

# delete the data generated suring the procedure
def files(ext = '*.mat'):
    for i in glob.glob(os.path.join(ext)):
        yield i
def remove_files(show = False):
    for i in files():
        if show:
            print(i)
        os.remove(i)
# models for comparing classifers
class Model:
    def __init__(self,model,name,x_src,y_src,x_tar,y_tar):
        self.x_src = x_src
        self.y_src = y_src
        self.x_tar = x_tar
        self.y_tar = y_tar
        self.model = model.fit(x_src,y_src)
        self.name = name
        self.predict_label_src = model.predict(self.x_src)
        self.predict_label_tar = model.predict(self.x_tar)
        self.acc_src = sklearn.metrics.accuracy_score(self.y_src, self.predict_label_src)
        self.acc_tar = sklearn.metrics.accuracy_score(self.y_tar, self.predict_label_tar)
#         self.predict_label_src_proba = model.predict_proba(self.x_src)
#         self.predict_label_tar_proba = model.predict_proba(self.x_tar)
        
class Models:
    def __init__(self,x_src,y_src,x_tar,y_tar):
        self.x_src = x_src
        self.y_src = y_src
        self.x_tar = x_tar
        self.y_tar = y_tar
        self.model = []
        
    def add(self,model,name):
        if (model in self.model):
            print("This model already exist!")
        else:
            self.model.append(Model(model,name,self.x_src,self.y_src,self.x_tar,self.y_tar))
    
    def show_src(self):
        print("Prediction on Source:\n",
              "%-8s"%("Model"),
              "%-18s"%("ACC"),
              "%-25s"%("Predicted class"),
              "%-25s"%("Numbers"))
        for i in range(len(self.model)):
            print("%-8s"%self.model[i].name,
                  "%-10.5f"%self.model[i].acc_src,
                  "%-25s"%np.unique(self.model[i].predict_label_src.reshape((-1,1)),return_counts=True)[0],
                  "%-25s"%np.unique(self.model[i].predict_label_src.reshape((-1,1)),return_counts=True)[1])
        print()
        
    def show_tar(self):  
        print("Prediction on Target:\n",
              "%-8s"%("Model"),
              "%-18s"%("ACC"),
              "%-25s"%("Predicted class"),
              "%-25s"%("Numbers"))
        for i in range(len(self.model)):
            print("%-8s"%self.model[i].name,
                  "%-10.5f"%self.model[i].acc_tar,
                  "%-25s"%np.unique(self.model[i].predict_label_tar.reshape((-1,1)),return_counts=True)[0],
                  "%-25s"%np.unique(self.model[i].predict_label_tar.reshape((-1,1)),return_counts=True)[1])
        print()
    
    def sort_tar(self,show=True):
        self.model.sort(key = lambda x:x.acc_tar, reverse=True)
        if (show == True):
            print("Prediction on Target:\n",
                  "%-8s"%("Model"),
                  "%-18s"%("ACC"),
                  "%-25s"%("Predicted class"),
                  "%-25s"%("Numbers"))
            for i in range(len(self.model)):
                print("%-8s"%self.model[i].name,
                      "%-10.5f"%self.model[i].acc_tar,
                      "%-25s"%np.unique(self.model[i].predict_label_tar.reshape((-1,1)),return_counts=True)[0],
                      "%-25s"%np.unique(self.model[i].predict_label_tar.reshape((-1,1)),return_counts=True)[1])
            print()

    def sort_src(self,show=True):
        self.model.sort(key = lambda x:x.acc_src, reverse=True)
        if (show == True):
            print("Prediction on Source:\n",
                  "%-8s"%("Model"),
                  "%-18s"%("ACC"),
                  "%-25s"%("Predicted class"),
                  "%-25s"%("Numbers"))
            for i in range(len(self.model)):
                print("%-8s"%self.model[i].name,
                      "%-10.5f"%self.model[i].acc_src,
                      "%-25s"%np.unique(self.model[i].predict_label_src.reshape((-1,1)),return_counts=True)[0],
                      "%-25s"%np.unique(self.model[i].predict_label_src.reshape((-1,1)),return_counts=True)[1])
            print()

# get the source and target data, balance the source data if 'balance' equal to True
def getdata(x_src, y_src, x_tar, y_tar, balance=False, z_score=False, test_size=0):   		
    if (balance == True):
        sm = SMOTE(random_state=42)
        x_src, y_src = sm.fit_sample(x_src, y_src)
        y_src = y_src.reshape((-1,1))
    if (test_size != 0):
        x_tar_train, x_tar_test, y_tar_train, y_tar_test = train_test_split(x_tar, y_tar, test_size=test_size)
    if (z_score == True):
        if (test_size != 0):
            scaler_src = preprocessing.StandardScaler().fit(x_src)
            scaler_tar = preprocessing.StandardScaler().fit(x_tar)
            x_src = scaler_src.transform(x_src)
            x_tar_train = scaler_tar.transform(x_tar_train)
            x_tar_test = scaler_tar.transform(x_tar_test)
            return x_src, y_src, x_tar_train, y_tar_train, x_tar_test, y_tar_test 
        else:
            scaler_src = preprocessing.StandardScaler().fit(x_src)
            scaler_tar = preprocessing.StandardScaler().fit(x_tar)
            x_src = scaler_src.transform(x_src)
            x_tar = scaler_tar.transform(x_tar)
            return x_src, y_src, x_tar, y_tar
    else:
        if (test_size != 0):
            return x_src, y_src, x_tar_train, y_tar_train, x_tar_test, y_tar_test 
        else:
            return x_src, y_src, x_tar, y_tar
        
# voting
def voting(shape,a,b,c):
    label = np.empty(shape)
    for i in range(shape):
        if a.predict_label_tar[i] == b.predict_label_tar[i]:
            label[i] = a.predict_label_tar[i]
        elif a.predict_label_tar[i] == c.predict_label_tar[i]:
            label[i] = a.predict_label_tar[i]
        elif b.predict_label_tar[i] == c.predict_label_tar[i]:
            label[i] = b.predict_label_tar[i]
        else:
            label[i] = -1
    label = label.reshape((-1,1))
    return label