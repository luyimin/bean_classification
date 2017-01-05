# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
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

# I cropped out each stereo image into its own file.
# You'll have to download the images to run this for yourself
#bean_bad = cv2.imread('./test_data_labeled/B_0_249/Bad/006_0011.jpg')
#bean_good = cv2.imread('./test_data_labeled/B_0_249/Good/001_0022.jpg')

def show_rgb_img(img):
    """Convenience function to display a typical color image"""
    return plt.imshow(cv2.cvtColor(img, cv2.CV_32S))

#show_rgb_img(bean_bad);    
#show_rgb_img(bean_good);

def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray

#bean_bad_gray = to_gray(bean_bad)
#bean_good_gray = to_gray(bean_good)

#plt.imshow(bean_bad_gray, cmap='gray');
#plt.imshow(bean_good_gray, cmap='gray');

def gen_sift_features(gray_img):
    sift = cv2.xfeatures2d.SIFT_create()
    # kp is the keypoints
    #
    # desc is the SIFT descriptors, they're 128-dimensional vectors
    # that we can use for our final features
    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc

def show_sift_features(gray_img, color_img, kp):
    return plt.imshow(cv2.drawKeypoints(gray_img, kp, color_img.copy()))

# generate SIFT keypoints and descriptors
#bean_bad_kp, bean_bad_desc = gen_sift_features(bean_bad_gray)
#bean_good_kp, bean_good_desc = gen_sift_features(bean_good_gray)

#print（'Here are what our SIFT features look like for the front-view octopus image:'）
#show_sift_features(bean_bad_gray, bean_bad, bean_bad_kp)
#show_sift_features(bean_good_gray, bean_good, bean_good_kp)


km_model = km(n_clusters=32)

current_array=np.reshape(np.arange(0,128,1),(1,128))

for dirPath, dirNames, fileNames in os.walk("./train_data_t/Bad"):
    for f in fileNames:
        if 'Store' not in f:
            file_path = os.path.join(dirPath, f)
            bean_train = cv2.imread(file_path)
            bean_train_gray = to_gray(bean_train)
            bean_train_kp, bean_train_desc = gen_sift_features(bean_train_gray)
            if bean_train_desc != None:
                current_array=np.concatenate((current_array,bean_train_desc),axis=0)
                
for dirPath, dirNames, fileNames in os.walk("./train_data_t/good"):
    for f in fileNames:
        if 'Store' not in f:
            file_path = os.path.join(dirPath, f)
            bean_train = cv2.imread(file_path)
            bean_train_gray = to_gray(bean_train)
            bean_train_kp, bean_train_desc = gen_sift_features(bean_train_gray)
            if bean_train_desc != None:
                current_array=np.concatenate((current_array,bean_train_desc),axis=0)
                

km_model.fit(current_array)

feature=[]
for dirPath, dirNames, fileNames in os.walk("./train_data_t/Bad"):
    for f in fileNames:
        if 'Store' not in f:
            file_path = os.path.join(dirPath, f)
            bean_train = cv2.imread(file_path)
            bean_train_gray = to_gray(bean_train)
            bean_train_kp, bean_train_desc = gen_sift_features(bean_train_gray)
            if bean_train_desc == None:
                print(file_path + "can't extract feature")
                feature.append([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            if bean_train_desc != None:
                desc_encode = km_model.predict(bean_train_desc)
                feature.append(np.histogram(desc_encode,range=(0,32),bins=32,normed=True)[0].tolist() + [0.0])
                
                
for dirPath, dirNames, fileNames in os.walk("./train_data_t/good"):
    for f in fileNames:
        if 'Store' not in f:
            file_path = os.path.join(dirPath, f)
            bean_train = cv2.imread(file_path)
            bean_train_gray = to_gray(bean_train)
            bean_train_kp, bean_train_desc = gen_sift_features(bean_train_gray)
            if bean_train_desc == None:
                print(file_path + "can't extract feature")
                feature.append([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0])
            if bean_train_desc != None:
                desc_encode = km_model.predict(bean_train_desc)
                feature.append(np.histogram(desc_encode,range=(0,32),bins=32,normed=True)[0].tolist() + [1.0])

x,y = [],[]
for i in feature:
    if i[:-1] != [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]:
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

