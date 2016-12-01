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

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict
print("# Tuning hyper-parameters for knn classfiication")
print()
param_grid = [
  {'n_neighbors': [5,10,15]},
  ]


clf = GridSearchCV(KNeighborsClassifier(n_neighbors=5,p=1),param_grid,cv=2,verbose=3)


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
	

	print('X_train',X_train.shape)
	print('X_test',X_test.shape)
	print('Y_train',Y_train.shape)
	print('Y_test',Y_test.shape)
	clf.fit(X_train, Y_train)


joblib.dump(clf.best_estimator_, 'knn_l1.model.pkl', compress = True)
print("The score on the best value of C: ",clf.score(X_test, Y_test))
print("scores: ",clf.grid_scores_)



n_neighbors = 15

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
y = iris.target

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

plt.show()