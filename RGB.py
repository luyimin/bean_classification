import matplotlib.pyplot as plt
import cv2
from skimage.feature import hog
from skimage import data, color, exposure
import os
import numpy as np
from skimage import data, io, filters
from sklearn.svm import SVC
from sklearn import cross_validation



feature = []
for dirPath, dirNames, fileNames in os.walk("./train_data_t/Bad"):

    for f in fileNames:
        if 'Store' not in f:
            file_path = os.path.join(dirPath, f)
            bean_train = io.imread(file_path)
            r= np.zeros((3, 256)) #[R, G, B]

            for i, row in enumerate(bean_train):
                for j, pixel in enumerate(row):
                    if pixel[0]<250: #濾掉白點
                        #print(i, j, pixel) #pixel: [R, G, B]
                        r[0][pixel[0]]+=1
                        r[1][pixel[1]]+=1
                        r[2][pixel[2]]+=1
            rcd = r.reshape((1,768)) / 255
            feature.append(rcd[0].tolist() + [0.0])

for dirPath, dirNames, fileNames in os.walk("./train_data_t/Good"):
#    RGB_count = dict()
    for f in fileNames:
        if 'Store' not in f:
            file_path = os.path.join(dirPath, f)
            bean_train = io.imread(file_path)
            r= np.zeros((3, 256)) #[R, G, B]

            for i, row in enumerate(bean_train):
                for j, pixel in enumerate(row):
                    if pixel[0]<250: #濾掉白點
                        #print(i, j, pixel) #pixel: [R, G, B]
                        r[0][pixel[0]]+=1
                        r[1][pixel[1]]+=1
                        r[2][pixel[2]]+=1
                        #clip[i][j] = pixel[::-1] #RGB2BGR
            rcd = r.reshape((1,768)) / 255
            feature.append(rcd[0].tolist() + [1.0])
    

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

unlabel_feature = []
path_list = []

for dirPath, dirNames, fileNames in os.walk("./Brazil_Cerrado_sprase_B"):

    for f in fileNames:
        if 'Store' not in f:
            file_path = os.path.join(dirPath, f)
            bean_train = io.imread(file_path)
            path_list.append(file_path)
            r= np.zeros((3, 256)) #[R, G, B]

            for i, row in enumerate(bean_train):
                for j, pixel in enumerate(row):
                    if pixel[0]<250: #濾掉白點
                        #print(i, j, pixel) #pixel: [R, G, B]
                        r[0][pixel[0]]+=1
                        r[1][pixel[1]]+=1
                        r[2][pixel[2]]+=1
            rcd = r.reshape((1,768)) / 255
            unlabel_feature.append(rcd[0].tolist())

unlabel_ans = []

clf = SVC(kernel='linear')
clf.fit(x, y)
test_unlable = clf.predict(unlabel_feature)
unlabel_ans.append(test_unlable)



file_re=zip(path_list, unlabel_ans )
