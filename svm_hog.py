import csv
import cv2

import sys
import os
import time
import string
import random
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix

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
from sklearn.neural_network import MLPClassifier


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



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



param_grid = [
  {'learning_rate_init': [0.0001,0.001,0.01,0.1]},
  ]

print("# Tuning hyper-parameters for multiclass(linear)")
print()
#clf = GridSearchCV(LinearSVC(C=1), param_grid, cv=3,verbose=3)
clf = GridSearchCV(MLPClassifier(solver='sgd', learning_rate_init=0.1,activation='relu',
                    hidden_layer_sizes=(200,100,50,25), random_state=1,max_iter=100),param_grid,cv=2,verbose=3)	

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

# print('X_train',X_train)
# print('X_test',X_test)
# print('Y_train',Y_train)
# print('Y_test',Y_test)


clf.fit(X_train, Y_train)


joblib.dump(clf.best_estimator_, 'hog_NN_relu.model.pkl', compress = True)
#clf=joblib.load('hog_NN_tanh.model.pkl')
print("The score on the best value of C: ",clf.score(X_test, Y_test))
#print("scores: ",clf.grid_scores_)

classes=[0,1,2,3,4,5,6,7,8,9]

Y_pred = clf.predict(X_test)

Y_pred=Y_pred.flatten()
Y_pred=np.array(Y_pred)

cnf_matrix = confusion_matrix(Y_test, Y_pred)



plt.figure()
plot_confusion_matrix(cnf_matrix, classes=classes,
                      title='Confusion matrix, without normalization')

plt.show()

