import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.tree import DecisionTreeClassifier
np.set_printoptions(precision=2)

#fashion = pd.read_table('fashion_quarter.csv',sep=',')
fashion = pd.read_table('usps_1000.csv',sep=',')
'''
0 T-shirt/top
1 Trouser
2 Pullover
3 Dress
4 Coat
5 Sandal
6 Shirt
7 Sneaker
8 Bag
9 Ankle boot
'''

target_names_fashion = ['T-shirt/top' ,'Trouser' ,'Pullover' ,'Dress' ,'Coat' ,'Sandal' ,'Shirt' ,'Sneaker' ,'Bag' ,'Ankle boot']
feature_names_fashion = list(fashion.columns.values)[:-1]
X_fashion = fashion[feature_names_fashion]
#y_fashion = fashion['class']
y_fashion = fashion['int0']
X_train_fashion, X_test_fashion, y_train_fashion, y_test_fashion = train_test_split(X_fashion, y_fashion, random_state=0)

#runwalk dataset

runwalk = pd.read_table('runwalk.csv',sep=',')
'''
0 Walking
1 Running 
'''

target_names_runwalk = ['Walking' ,'Running']
feature_names_runwalk = list(runwalk.columns.values)[:-1]
X_runwalk = runwalk[feature_names_runwalk]
y_runwalk = runwalk['activity']
X_train_runwalk, X_test_runwalk, y_train_runwalk, y_test_runwalk = train_test_split(X_runwalk, y_runwalk, random_state=0)

####################
#K-Nearest Neighbors
####################


from sklearn.preprocessing import MinMaxScaler

'''
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
# we must apply the scaling to the test set that we computed for the training set
X_test_scaled = scaler.transform(X_test)

'''
####################
# Learning  curve
####################

from sklearn.model_selection import learning_curve

# fashion dataset
# -----------

train_sizes, train_scores, test_scores = learning_curve(KNeighborsClassifier(n_neighbors = 17), X_train_fashion, y_train_fashion, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)

print "train sizes", train_sizes 
print "train scores", train_scores 
print "test scores", test_scores

fig = plt.figure()
plt.title("K-Nearest Neighbors Learning Curves (K = 17)  | fashion dataset")
plt.xlabel("Training examples")
plt.ylabel("Accuracy (% Correctly Labeled)")

train_scores_mean = np.mean(train_scores, axis=1) * 100
train_scores_std = np.std(train_scores, axis=1) * 100
test_scores_mean = np.mean(test_scores, axis=1) * 100
test_scores_std = np.std(test_scores, axis=1) * 100
train_scores = train_scores * 100
test_scores = test_scores * 100
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

plt.legend(loc="best")
#plt.show()
fig.savefig('knn_fashion_lc.png')

# RunWalk Learning Curve
# ---------

from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(KNeighborsClassifier(n_neighbors = 3), X_train_runwalk, y_train_runwalk, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)

print "train sizes", train_sizes 
print "train scores", train_scores 
print "test scores", test_scores

fig = plt.figure()
plt.title("K-Nearest Neighbors Learning Curves (k = 3) | runwalk dataset")
plt.xlabel("Training examples")
plt.ylabel("Accuracy (% Correctly Labeled)")

train_scores_mean = np.mean(train_scores, axis=1) * 100
train_scores_std = np.std(train_scores, axis=1) * 100
test_scores_mean = np.mean(test_scores, axis=1) * 100
test_scores_std = np.std(test_scores, axis=1) * 100
train_scores = train_scores * 100
test_scores = test_scores * 100
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

plt.legend(loc="best")
#plt.show()
fig.savefig('knn_rw_lc.png')


####################
# Cross-validation
####################

'''
for num_k in range(1,30):
   knn = KNeighborsClassifier(n_neighbors = num_k)
   #knn.fit(X_train_scaled, y_train)
   knn.fit(X_train, y_train)
   #print('Accuracy of K-NN classifier on training set: {:.2f}'
   #     .format(knn.score(X_train_scaled, y_train)))
   print "K-NN where k =", num_k
   print('Accuracy of K-NN classifier on training set: {:.2f}'
        .format(knn.score(X_train, y_train)))
   #print('Accuracy of K-NN classifier on test set: {:.2f}'
   #     .format(knn.score(X_test_scaled, y_test)))
   print('Accuracy of K-NN classifier on test set: {:.2f}'
        .format(knn.score(X_test, y_test)))
'''   
from sklearn.model_selection import cross_val_score
'''
clf = KNeighborsClassifier(n_neighbors = 14)
#X = X_fruits_2d.as_matrix()
#y = y_fruits_2d.as_matrix()
#cv_scores = cross_val_score(clf, X, y)
cv_scores = cross_val_score(clf, X_train, y_train, cv = 10)

print('Cross-validation scores (10-fold):', cv_scores)
print('Mean cross-validation score (10-fold): {:.3f}'
     .format(np.mean(cv_scores)))
'''
####################
#Validation curve 
####################

from sklearn.model_selection import validation_curve

param_range = np.linspace(1, 20, 20, dtype=int)
print param_range
train_scores, test_scores = validation_curve(KNeighborsClassifier(), X_train_fashion, y_train_fashion,
                                            param_name='n_neighbors',
                                            param_range=param_range, cv=5)

#print(train_scores)
#print(test_scores)

#  See:  http://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html
fig = plt.figure()

train_scores_mean = np.mean(train_scores, axis=1) * 100
train_scores_std = np.std(train_scores, axis=1) * 100
test_scores_mean = np.mean(test_scores, axis=1) * 100
test_scores_std = np.std(test_scores, axis=1) * 100
train_scores = train_scores * 100
test_scores = test_scores * 100

plt.title('K-Nearest Neighbors Model Complexity for Number of Neighbors | fashion dataset')
plt.xlabel('Number of Neighbors')
plt.ylabel("Accuracy (% Correctly Labeled)")
plt.grid(True)
plt.ylim(65, 101)
plt.xlim(0,param_range[-1])
plt.xticks(param_range)
lw = 2

plt.plot(param_range, train_scores_mean, label='Training score',
            color='darkorange', lw=lw)

plt.fill_between(param_range, train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std, alpha=0.2,
                color='darkorange', lw=lw)

plt.plot(param_range, test_scores_mean, label='Cross-validation score',
            color='navy', lw=lw)

plt.fill_between(param_range, test_scores_mean - test_scores_std,
                test_scores_mean + test_scores_std, alpha=0.2,
                color='navy', lw=lw)

plt.legend(loc='best')
#plt.show()
fig.savefig('knn_fash_mc.png')

# Model_complexity: runwalk
# --------
param_range = np.linspace(1, 20, 20, dtype=int)
print param_range
train_scores, test_scores = validation_curve(KNeighborsClassifier(), X_train_runwalk, y_train_runwalk,
                                            param_name='n_neighbors',
                                            param_range=param_range, cv=5)

print(train_scores)
print(test_scores)

#  See:  http://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html
fig = plt.figure()

train_scores_mean = np.mean(train_scores, axis=1) * 100
train_scores_std = np.std(train_scores, axis=1) * 100
test_scores_mean = np.mean(test_scores, axis=1) * 100
test_scores_std = np.std(test_scores, axis=1) * 100
train_scores = train_scores * 100
test_scores = test_scores * 100

plt.title('K-Nearest Neighbors Model Complexity for Number of Neighbors | runwalk dataset')
plt.xlabel('Number of Neighbors')
plt.ylabel("Accuracy (% Correctly Labeled)")
plt.grid(True)
plt.ylim(65, 101)
plt.xlim(0,param_range[-1])
plt.xticks(param_range)
lw = 2

plt.plot(param_range, train_scores_mean, label='Training score',
            color='darkorange', lw=lw)

plt.fill_between(param_range, train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std, alpha=0.2,
                color='darkorange', lw=lw)

plt.plot(param_range, test_scores_mean, label='Cross-validation score',
            color='navy', lw=lw)

plt.fill_between(param_range, test_scores_mean - test_scores_std,
                test_scores_mean + test_scores_std, alpha=0.2,
                color='navy', lw=lw)

plt.legend(loc='best')
#plt.show()
fig.savefig('knn_wc_mc.png')

## Testset performance
clf2 = KNeighborsClassifier(n_neighbors = 17, random_state = 0).fit(X_train_fashion, y_train_fashion)
print('fashion | Accuracy of K-NN classifier on training set | num neighbors = 17  : {:.2f}'
     .format(clf2.score(X_train_fashion, y_train_fashion)))
print('fashion | Accuracy of K-NN classifier on test set | num neighbors = 17 : {:.2f}'
     .format(clf2.score(X_test_fashion, y_test_fashion)))

clf = KNeighborsClassifier(n_neighbors = 3, random_state = 0).fit(X_train_runwalk, y_train_runwalk)
print('runwalk | Accuracy of K-NN classifier on training set | num neightbors = 3 : {:.2f}'
     .format(clf.score(X_train_runwalk, y_train_runwalk)))
print('runwalk | Accuracy of K-NN  classifier on test set | num neighbors = 3 : {:.2f}'
     .format(clf.score(X_test_runwalk, y_test_runwalk)))

