import matplotlib.pyplot as plt
import cv2
from skimage.feature import hog
from skimage import data, color, exposure
import os
import numpy as np

from sklearn.svm import SVC
from sklearn import cross_validation


feature = []
for dirPath, dirNames, fileNames in os.walk("./train_data_t/Bad"):
    for f in fileNames:
        if 'Store' not in f:
            file_path = os.path.join(dirPath, f)
            bean_train = cv2.imread(file_path)
            image = color.rgb2gray(bean_train)
            fd, hog_image = hog(image, orientations=4, pixels_per_cell=(8, 8),
                    cells_per_block=(1, 1), visualise=True) 
            feature.append(fd.tolist() + [0.0])

for dirPath, dirNames, fileNames in os.walk("./train_data_t/Good"):
    for f in fileNames:
        if 'Store' not in f:
            file_path = os.path.join(dirPath, f)
            bean_train = cv2.imread(file_path)
            image = color.rgb2gray(bean_train)
            fd, hog_image = hog(image, orientations=4, pixels_per_cell=(8, 8),
                    cells_per_block=(1, 1), visualise=True) 
            feature.append(fd.tolist() + [1.0])

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



#
#fd, hog_image = hog(image, orientations=4, pixels_per_cell=(8, 8),
#                    cells_per_block=(1, 1), visualise=True)
#
#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
#
#ax1.axis('off')
#ax1.imshow(image, cmap=plt.cm.gray)
#ax1.set_title('Input image')
#ax1.set_adjustable('box-forced')
#
## Rescale histogram for better display
#hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
#
#ax2.axis('off')
#ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
#ax2.set_title('Histogram of Oriented Gradients')
#ax1.set_adjustable('box-forced')
#plt.show()
