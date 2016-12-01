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

import itertools
from sklearn.neighbors import KNeighborsClassifier
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
	im = cv2.imread(X1[i])
	hog = cv2.HOGDescriptor((32,32), (16,16), (8,8), (8,8), 9)
	h = hog.compute(im)
	if(i<2400):
		X_train.append(h)
	else:
		X_test.append(h)




for i in range(3400):
	im = cv2.imread(X2[i])
	hog = cv2.HOGDescriptor((32,32), (16,16), (8,8), (8,8), 9)
	
	h = hog.compute(im)
	if(i<2400):
		X_train.append(h)
	else:
		X_test.append(h)
for i in range(3400):
	im = cv2.imread(X3[i])
	hog = cv2.HOGDescriptor((32,32), (16,16), (8,8), (8,8), 9)
	
	h = hog.compute(im)
	if(i<2400):
		X_train.append(h)
	else:
		X_test.append(h)
for i in range(3400):
	im = cv2.imread(X4[i])
	hog = cv2.HOGDescriptor((32,32), (16,16), (8,8), (8,8), 9)
	
	h = hog.compute(im)
	if(i<2400):
		X_train.append(h)
	else:
		X_test.append(h)
for i in range(3400):
	im = cv2.imread(X5[i])
	hog = cv2.HOGDescriptor((32,32), (16,16), (8,8), (8,8), 9)
	
	h = hog.compute(im)
	if(i<2400):
		X_train.append(h)
	else:
		X_test.append(h)
for i in range(3400):
	im = cv2.imread(X6[i])
	hog = cv2.HOGDescriptor((32,32), (16,16), (8,8), (8,8), 9)
	
	h = hog.compute(im)
	if(i<2400):
		X_train.append(h)
	else:
		X_test.append(h)
for i in range(3400):
	im = cv2.imread(X7[i])
	hog = cv2.HOGDescriptor((32,32), (16,16), (8,8), (8,8), 9)
	
	h = hog.compute(im)
	if(i<2400):
		X_train.append(h)
	else:
		X_test.append(h)
for i in range(3400):
	im = cv2.imread(X8[i])
	hog = cv2.HOGDescriptor((32,32), (16,16), (8,8), (8,8), 9)
	
	h = hog.compute(im)
	if(i<2400):
		X_train.append(h)
	else:
		X_test.append(h)
for i in range(3400):
	im = cv2.imread(X9[i])
	hog = cv2.HOGDescriptor((32,32), (16,16), (8,8), (8,8), 9)
	
	h = hog.compute(im)
	if(i<2400):
		X_train.append(h)
	else:
		X_test.append(h)
for i in range(3400):
	im = cv2.imread(X10[i])
	hog = cv2.HOGDescriptor((32,32), (16,16), (8,8), (8,8), 9)
	
	h = hog.compute(im)
	if(i<2400):
		X_train.append(h)
	else:
		X_test.append(h)

print("# Tuning hyper-parameters for knn classfiication")
print()

param_grid = [
  {'n_neighbors': [5,10,20]},
  ]


clf = GridSearchCV(KNeighborsClassifier(n_neighbors=5),param_grid,cv=2,verbose=3)
X_train=np.array(X_train)
X_test=np.array(X_test)

X_train=np.ravel(X_train)
X_train=np.reshape(X_train,(24000,324))

X_test=np.ravel(X_test)
X_test=np.reshape(X_test,(10000,324))

Y_train=np.array(Y_train)
Y_test=np.array(Y_test)


print('X_train',X_train.shape)
print('X_test',X_test.shape)
print('Y_train',Y_train.shape)
print('Y_test',Y_test.shape)

clf.fit(X_train, Y_train)


joblib.dump(clf.best_estimator_, 'knn_hog.model.pkl', compress = True)
print("The score on the best value of C: ",clf.score(X_test, Y_test))
print("scores: ",clf.grid_scores_)
