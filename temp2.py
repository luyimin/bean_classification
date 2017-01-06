# -*- coding: utf-8 -*-
#%matplotlib inline
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans as km
from sklearn.svm import SVC
from sklearn import cross_validation

print ('OpenCV Version (should be 3.1.0, with nonfree packages installed, for this tutorial):')
print (cv2.__version__)

def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray

feature=[]

for dirPath, dirNames, fileNames in os.walk("./train_data_t/Bad"):
    for f in fileNames:
        if 'Store' not in f:
            file_path = os.path.join(dirPath, f)
            bean_train = cv2.imread(file_path)
            bean_train_gray = to_gray(bean_train)
            rd = bean_train_gray.reshape((1,40000)) / 256 
            feature.append(rd[0].tolist()+[0.0])
                
            
for dirPath, dirNames, fileNames in os.walk("./train_data_t/good"):
    for f in fileNames:
        if 'Store' not in f:
            file_path = os.path.join(dirPath, f)
            bean_train = cv2.imread(file_path)
            bean_train_gray = to_gray(bean_train)
            rd = bean_train_gray.reshape((1,40000)) / 256 
            feature.append(rd[0].tolist()+[1.0])

x,y = [],[]
for i in feature:
    x.append(i[:-1])
    y.append(i[-1])
    
Label=[]
Ans=[]

x = np.asarray(x)
y = np.asarray(y)
loo = cross_validation.LeaveOneOut(x.shape[0])

for train_i, test_i in loo:
    clf = SVC(kernel='linear')
    clf.fit(x[train_i], y[train_i])
    testY=clf.predict(x[test_i])[0]
    Ans.append(testY)
    Label.append(y[test_i])
   # print('Sample %d score: %f' % (test_i[0], score))
from sklearn.metrics import confusion_matrix
Res=confusion_matrix(Label,Ans)
print(Res)
from sklearn.metrics import recall_score
print(recall_score(Label,Ans, average='binary'))

