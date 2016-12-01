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
from skimage.feature import ORB, match_descriptors


from skimage import data, color, exposure
from skimage import io
from skimage.transform import resize



import itertools

from skimage.feature import ORB, match_descriptors

Y_train=[1]*240+[2]*240+[3]*240+[4]*240+[5]*240+[6]*240+[7]*240+[8]*240+[9]*240+[10]*240
Y_test=[1]*100+[2]*100+[3]*100+[4]*100+[5]*100+[6]*100+[7]*100+[8]*100+[9]*100+[10]*100

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
		if(row[0]=='1' and len(X1)<=350):
			X1.append('train/'+str(counter)+'.png')
		elif(row[0]=='2' and len(X2)<=350):
			X2.append('train/'+str(counter)+'.png')
		elif(row[0]=='3' and len(X3)<=350):
			X3.append('train/'+str(counter)+'.png')
		elif(row[0]=='4' and len(X4)<=350):
			X4.append('train/'+str(counter)+'.png')
		elif(row[0]=='5' and len(X5)<=350):
			X5.append('train/'+str(counter)+'.png')
		elif(row[0]=='6' and len(X6)<=350):
			X6.append('train/'+str(counter)+'.png')
		elif(row[0]=='7' and len(X7)<=350):
			X7.append('train/'+str(counter)+'.png')
		
		elif(row[0]=='8' and len(X8)<=350):
			X8.append('train/'+str(counter)+'.png')
		elif(row[0]=='9' and len(X9)<=350):
			X9.append('train/'+str(counter)+'.png')
		elif(row[0]=='10' and len(X10)<=350):
			X10.append('train/'+str(counter)+'.png')
		if(counter==10000):
			break

#print(X1)
print("lenght",len(X1))

for i in range(340):
	img = cv2.imread(X1[i])
	img = color.rgb2gray(img)
	img=resize(img,(128,128))
	img1 = np.array(img)
	img1=img1*255

	detector_extractor1 = ORB(downscale=1.2, n_keypoints=100)
	detector_extractor1.detect_and_extract(img1)
	des=detector_extractor1.keypoints

	if(i<240):
		X_train.append(des)
	else:
		X_test.append(des)


for i in range(340):
	img = cv2.imread(X2[i])
	img = color.rgb2gray(img)
	img=resize(img,(128,128))
	img1 = np.array(img)
	img1=img1*255

	detector_extractor1 = ORB(downscale=1.2, n_keypoints=100)
	detector_extractor1.detect_and_extract(img1)
	h=detector_extractor1.keypoints
	if(i<240):
		X_train.append(h)
	else:
		X_test.append(h)
for i in range(340):
	img = cv2.imread(X3[i])
	img = color.rgb2gray(img)
	img=resize(img,(128,128))
	img1 = np.array(img)
	img1=img1*255

	detector_extractor1 = ORB(downscale=1.2, n_keypoints=100)
	detector_extractor1.detect_and_extract(img1)
	h=detector_extractor1.keypoints
	if(i<240):
		X_train.append(h)
	else:
		X_test.append(h)
for i in range(340):
	img = cv2.imread(X4[i])
	img = color.rgb2gray(img)
	img=resize(img,(128,128))
	img1 = np.array(img)
	img1=img1*255

	detector_extractor1 = ORB(downscale=1.2, n_keypoints=100)
	detector_extractor1.detect_and_extract(img1)
	h=detector_extractor1.keypoints
	if(i<240):
		X_train.append(h)
	else:
		X_test.append(h)
for i in range(340):
	img = cv2.imread(X5[i])
	img = color.rgb2gray(img)
	img=resize(img,(128,128))
	img1 = np.array(img)
	img1=img1*255

	detector_extractor1 = ORB(downscale=1.2, n_keypoints=100)
	detector_extractor1.detect_and_extract(img1)
	h=detector_extractor1.keypoints
	if(i<240):
		X_train.append(h)
	else:
		X_test.append(h)
for i in range(340):
	img = cv2.imread(X6[i])
	img = color.rgb2gray(img)
	img=resize(img,(128,128))
	img1 = np.array(img)
	img1=img1*255

	detector_extractor1 = ORB(downscale=1.2, n_keypoints=100)
	detector_extractor1.detect_and_extract(img1)
	h=detector_extractor1.keypoints
	if(i<240):
		X_train.append(h)
	else:
		X_test.append(h)
for i in range(340):
	img = cv2.imread(X7[i])
	img = color.rgb2gray(img)
	img=resize(img,(128,128))
	img1 = np.array(img)
	img1=img1*255

	detector_extractor1 = ORB(downscale=1.2, n_keypoints=100)
	detector_extractor1.detect_and_extract(img1)
	h=detector_extractor1.keypoints
	if(i<240):
		X_train.append(h)
	else:
		X_test.append(h)
for i in range(340):
	img = cv2.imread(X8[i])
	img = color.rgb2gray(img)
	img=resize(img,(128,128))
	img1 = np.array(img)
	img1=img1*255

	detector_extractor1 = ORB(downscale=1.2, n_keypoints=100)
	detector_extractor1.detect_and_extract(img1)
	h=detector_extractor1.keypoints
	if(i<240):
		X_train.append(h)
	else:
		X_test.append(h)
for i in range(340):
	img = cv2.imread(X9[i])
	img = color.rgb2gray(img)
	img=resize(img,(128,128))
	img1 = np.array(img)
	img1=img1*255

	detector_extractor1 = ORB(downscale=1.2, n_keypoints=100)
	detector_extractor1.detect_and_extract(img1)
	h=detector_extractor1.keypoints
	if(i<240):
		X_train.append(h)
	else:
		X_test.append(h)
for i in range(340):
	img = cv2.imread(X10[i])
	img = color.rgb2gray(img)
	img=resize(img,(128,128))
	img1 = np.array(img)
	img1=img1*255

	detector_extractor1 = ORB(downscale=1.2, n_keypoints=100)
	detector_extractor1.detect_and_extract(img1)
	h=detector_extractor1.keypoints
	if(i<240):
		X_train.append(h)
	else:
		X_test.append(h)



param_grid = [
  {'C': [0.0001,0.001,0.01,0.1]},
  ]

print("# Tuning hyper-parameters for multiclass(linear)")
print()
clf = GridSearchCV(LinearSVC(C=1), param_grid, cv=3,verbose=3)
# clf = GridSearchCV(MLPClassifier(solver='sgd', learning_rate_init=0.1,activation='relu',
#                     hidden_layer_sizes=(200,100,50,25), random_state=1,max_iter=100),param_grid,cv=2,verbose=3)	

X_train=np.array(X_train)
X_test=np.array(X_test)

X_train=np.ravel(X_train)
X_train=np.reshape(X_train,(2400,200))

X_test=np.ravel(X_test)
X_test=np.reshape(X_test,(1000,200))

Y_train=np.array(Y_train)
Y_test=np.array(Y_test)


print('X_train',X_train.shape)
print('X_test',X_test.shape)
print('Y_train',Y_train.shape)
print('Y_test',Y_test.shape)

# print('X_train',X_train)
# print('X_test',X_test)
# print('Y_train',Y_train)
# print('Y_test',Y_test)


clf.fit(X_train, Y_train)


joblib.dump(clf.best_estimator_, 'orb_svm.model.pkl', compress = True)
print("The score on the best value of C: ",clf.score(X_test, Y_test))
print("scores: ",clf.grid_scores_)
