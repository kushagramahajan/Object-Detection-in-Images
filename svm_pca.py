from __future__ import print_function

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
from sklearn.decomposition import PCA




def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


################ multilabel classifiction


param_grid = [
  {'C': [0.000001,0.00001,0.0001,0.001,0.1]},
  ]

print("# Tuning hyper-parameters for multiclass(linear)")
print()
clf = GridSearchCV(LinearSVC(C=1), param_grid, cv=3,verbose=3)


###### test data
xs=[]
ys=[]

d = unpickle('data/cifar-10-batches-py/test_batch')
xs.append(d['data'])
ys.append(d['labels'])

x = np.concatenate(xs)/np.float32(255)
y = np.concatenate(ys)
x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
x = x.reshape((x.shape[0], 32, 32, 3)).transpose(0,3,1,2)

pixel_mean = np.mean(x[0:10000],axis=0)

x -= pixel_mean

X_test = x[0:10000,:,:,:]
Y_test = y[0:10000]
X_test=np.ravel(X_test)
X_test=np.reshape(X_test,(10000,3072))


for j in range(1):
	xs = []
	ys = []
	d = unpickle('data/cifar-10-batches-py/data_batch_'+`j+1`)
	x = d['data']
	y = d['labels']
	xs.append(x)
	ys.append(y)

	x = np.concatenate(xs)/np.float32(255)
	y = np.concatenate(ys)
	x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
	x = x.reshape((x.shape[0], 32, 32, 3)).transpose(0,3,1,2)

	# subtract per-pixel mean
	pixel_mean = np.mean(x[0:20000],axis=0)
	#pickle.dump(pixel_mean, open("cifar10-pixel_mean.pkl","wb"))
	x -= pixel_mean

	# create mirrored images
	X_train = x[0:10000,:,:,:]
	Y_train = y[0:10000]
	X_train=np.ravel(X_train)
	X_train=np.reshape(X_train,(10000,3072))
	pca = PCA(n_components=768)# adjust yourself
	pca.fit(X_train)
	X_train = pca.transform(X_train)
	X_test = pca.transform(X_test)

	print('X_train',X_train.shape)
	print('X_test',X_test.shape)
	print('Y_train',Y_train.shape)
	print('Y_test',Y_test.shape)
	clf.fit(X_train, Y_train)

joblib.dump(clf.best_estimator_, 'multi_pca_768.model.pkl', compress = True)
print("The score on the best value of C: ",clf.score(X_test, Y_test))
print("scores: ",clf.grid_scores_)
	