import csv
import cv2

import sys
import os
import time
import string
import random
import pickle
import numpy as np

import sklearn
from matplotlib import pylab as plt
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from roc import generate_roc



from skimage import data, color, exposure
from skimage import io
from skimage.transform import resize

from skimage.feature import daisy
from skimage import data
import matplotlib.pyplot as plt


Y_train=[1]*2400+[2]*2400+[3]*2400+[4]*2400+[5]*2400+[6]*2400+[7]*2400+[8]*2400+[9]*2400+[10]*2400
Y_test=[1]*1000+[2]*1000+[3]*1000+[4]*1000+[5]*1000+[6]*1000+[7]*1000+[8]*1000+[9]*1000+[10]*1000

counter=0

X1=[]
X2=[]
X3=[]
X4=[]
X5=[]
X6=[]
X7=[]
X8=[]
X9=[]
X10=[]
X_train=[]
X_test=[]

with open('trainLabels.csv', 'rb') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
	for row in spamreader:
		counter+=1
		if(row[0]=='1' and len(X1)<=4000):
			X1.append('train/'+str(counter)+'.png')
		elif(row[0]=='2' and len(X2)<=4000):
			X2.append('train/'+str(counter)+'.png')
		elif(row[0]=='3' and len(X3)<=4000):
			X3.append('train/'+str(counter)+'.png')
		elif(row[0]=='4' and len(X4)<=4000):
			X4.append('train/'+str(counter)+'.png')
		elif(row[0]=='5' and len(X5)<=4000):
			X5.append('train/'+str(counter)+'.png')
		elif(row[0]=='6' and len(X6)<=4000):
			X6.append('train/'+str(counter)+'.png')
		elif(row[0]=='7' and len(X7)<=4000):
			X7.append('train/'+str(counter)+'.png')
		
		elif(row[0]=='8' and len(X8)<=4000):
			X8.append('train/'+str(counter)+'.png')
		elif(row[0]=='9' and len(X9)<=4000):
			X9.append('train/'+str(counter)+'.png')
		elif(row[0]=='10' and len(X10)<=4000):
			X10.append('train/'+str(counter)+'.png')
		else:
			break			



#print(X1)
print("lenght",len(X1))

for i in range(3400):
	img = cv2.imread(X1[i])
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	orb = cv2.ORB()
	kp = orb.detect(img,None)
	print("kp",kp)
	
	# compute the descriptors with ORB
	kp, des = orb.compute(img, kp)
	print("des",des)

	# draw only keypoints location,not size and orientation
	img2 = cv2.drawKeypoints(img,kp,color=(0,255,0), flags=0)
	plt.imshow(img2),plt.show()
	break

	# fast = cv2.FastFeatureDetector()
	# # find the keypoints with ORB
	# kp = fast.detect(im1,None)
	# im2 = cv2.drawKeypoints(im1, kp, color=(255,0,0))

	# # compute the descriptors with ORB
	# des = orb.compute(im2, kp)
	# print('kp: ',len(kp))
	# print('des: ',des)
	# if(i==10):
	# 	break
	# if(i<2400):
	# 	X_train.append(des)
	# else:
	# 	X_test.append(des)




# for i in range(3400):
# 	im = cv2.imread(X2[i])
# 	hog = cv2.HOGDescriptor((32,32), (16,16), (8,8), (8,8), 9)
	
# 	h = hog.compute(im)
# 	if(i<2400):
# 		X_train.append(h)
# 	else:
# 		X_test.append(h)
# for i in range(3400):
# 	im = cv2.imread(X3[i])
# 	hog = cv2.HOGDescriptor((32,32), (16,16), (8,8), (8,8), 9)
	
# 	h = hog.compute(im)
# 	if(i<2400):
# 		X_train.append(h)
# 	else:
# 		X_test.append(h)
# for i in range(3400):
# 	im = cv2.imread(X4[i])
# 	hog = cv2.HOGDescriptor((32,32), (16,16), (8,8), (8,8), 9)
	
# 	h = hog.compute(im)
# 	if(i<2400):
# 		X_train.append(h)
# 	else:
# 		X_test.append(h)
# for i in range(3400):
# 	im = cv2.imread(X5[i])
# 	hog = cv2.HOGDescriptor((32,32), (16,16), (8,8), (8,8), 9)
	
# 	h = hog.compute(im)
# 	if(i<2400):
# 		X_train.append(h)
# 	else:
# 		X_test.append(h)
# for i in range(3400):
# 	im = cv2.imread(X6[i])
# 	hog = cv2.HOGDescriptor((32,32), (16,16), (8,8), (8,8), 9)
	
# 	h = hog.compute(im)
# 	if(i<2400):
# 		X_train.append(h)
# 	else:
# 		X_test.append(h)
# for i in range(3400):
# 	im = cv2.imread(X7[i])
# 	hog = cv2.HOGDescriptor((32,32), (16,16), (8,8), (8,8), 9)
	
# 	h = hog.compute(im)
# 	if(i<2400):
# 		X_train.append(h)
# 	else:
# 		X_test.append(h)
# for i in range(3400):
# 	im = cv2.imread(X8[i])
# 	hog = cv2.HOGDescriptor((32,32), (16,16), (8,8), (8,8), 9)
	
# 	h = hog.compute(im)
# 	if(i<2400):
# 		X_train.append(h)
# 	else:
# 		X_test.append(h)
# for i in range(3400):
# 	im = cv2.imread(X9[i])
# 	hog = cv2.HOGDescriptor((32,32), (16,16), (8,8), (8,8), 9)
	
# 	h = hog.compute(im)
# 	if(i<2400):
# 		X_train.append(h)
# 	else:
# 		X_test.append(h)
# for i in range(3400):
# 	im = cv2.imread(X10[i])
# 	hog = cv2.HOGDescriptor((32,32), (16,16), (8,8), (8,8), 9)
	
# 	h = hog.compute(im)
# 	if(i<2400):
# 		X_train.append(h)
# 	else:
# 		X_test.append(h)



# param_grid = [
#   {'C': [0.000001,0.00001,0.0001,0.001,0.1,1,10]},
#   ]

# print("# Tuning hyper-parameters for multiclass(linear)")
# print()
# clf = GridSearchCV(LinearSVC(C=1), param_grid, cv=3,verbose=3)

# X_train=np.array(X_train)
# X_test=np.array(X_test)

# X_train=np.ravel(X_train)
# X_train=np.reshape(X_train,(24000,324))

# X_test=np.ravel(X_test)
# X_test=np.reshape(X_test,(10000,324))

# Y_train=np.array(Y_train)
# Y_test=np.array(Y_test)


# print('X_train',X_train.shape)
# print('X_test',X_test.shape)
# print('Y_train',Y_train.shape)
# print('Y_test',Y_test.shape)

# # print('X_train',X_train)
# # print('X_test',X_test)
# # print('Y_train',Y_train)
# # print('Y_test',Y_test)


# clf.fit(X_train, Y_train)


# joblib.dump(clf.best_estimator_, 'sift_svm.model.pkl', compress = True)
# print("The score on the best value of C: ",clf.score(X_test, Y_test))
# print("scores: ",clf.grid_scores_)
