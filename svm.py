import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
from sklearn.model_selection import train_test_split

np.set_printoptions(precision=2)

fashion = pd.read_table('usps.csv',sep=',')
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

#target_names_fashion = ['T-shirt/top' ,'Trouser' ,'Pullover' ,'Dress' ,'Coat' ,'Sandal' ,'Shirt' ,'Sneaker' ,'Bag' ,'Ankle boot']
#feature_names_fashion = list(fashion.columns.values)[:-1]
X_fashion = fashion[feature_names_fashion]
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



from sklearn.preprocessing import MinMaxScaler

'''
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
# we must apply the scaling to the test set that we computed for the training set
X_test_scaled = scaler.transform(X_test)

'''

####################
#Support Vector Machines
####################
#Linear Support Vector Machine

from sklearn.svm import SVC
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot

'''
X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2, random_state = 0)

fig, subaxes = plt.subplots(1, 1, figsize=(7, 5))
this_C = 1.0
clf = SVC(kernel = 'linear', C=this_C).fit(X_train, y_train)
title = 'Linear SVC, C = {:.3f}'.format(this_C)
plot_class_regions_for_classifier_subplot(clf, X_train, y_train, None, None, title, subaxes)

#Linear Support Vector Machine: C parameter

from sklearn.svm import LinearSVC
from adspy_shared_utilities import plot_class_regions_for_classifier

X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2, random_state = 0)
fig, subaxes = plt.subplots(1, 2, figsize=(8, 4))

for this_C, subplot in zip([0.00001, 100], subaxes):
    clf = LinearSVC(C=this_C).fit(X_train, y_train)
    title = 'Linear SVC, C = {:.5f}'.format(this_C)
    plot_class_regions_for_classifier_subplot(clf, X_train, y_train,
                                             None, None, title, subplot)
plt.tight_layout()
'''

from sklearn.svm import LinearSVC
#X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)

clf = LinearSVC().fit(X_train, y_train)
print('fashion dataset')
print('Accuracy of Linear SVC classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Linear SVC classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))

####################
##Multi-class classification with linear models
####################

#LinearSVC with M classes generates M one vs rest classifiers.
'''
from sklearn.svm import LinearSVC

X_train, X_test, y_train, y_test = train_test_split(X_fruits_2d, y_fruits_2d, random_state = 0)

clf = LinearSVC(C=5, random_state = 67).fit(X_train, y_train)
print('Coefficients:\n', clf.coef_)
print('Intercepts:\n', clf.intercept_)
'''
#Multi-class results on the fruit dataset
'''
plt.figure(figsize=(6,6))
colors = ['r', 'g', 'b', 'y']
cmap_fruits = ListedColormap(['#FF0000', '#00FF00', '#0000FF','#FFFF00'])

plt.scatter(X_fruits_2d[['height']], X_fruits_2d[['width']],
           c=y_fruits_2d, cmap=cmap_fruits, edgecolor = 'black', alpha=.7)

x_0_range = np.linspace(-10, 15)

for w, b, color in zip(clf.coef_, clf.intercept_, ['r', 'g', 'b', 'y']):
    # Since class prediction with a linear model uses the formula y = w_0 x_0 + w_1 x_1 + b, 
    # and the decision boundary is defined as being all points with y = 0, to plot x_1 as a 
    # function of x_0 we just solve w_0 x_0 + w_1 x_1 + b = 0 for x_1:
    plt.plot(x_0_range, -(x_0_range * w[0] + b) / w[1], c=color, alpha=.8)
    
plt.legend(target_names_fruits)
plt.xlabel('height')
plt.ylabel('width')
plt.xlim(-2, 12)
plt.ylim(-2, 15)
plt.show()
'''

####################
# Kernelized Support Vector Machines
####################
# Classification

from sklearn.svm import SVC
from adspy_shared_utilities import plot_class_regions_for_classifier
'''
X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state = 0)

# The default SVC kernel is radial basis function (RBF)
plot_class_regions_for_classifier(SVC().fit(X_train, y_train),
                                 X_train, y_train, None, None,
                                 'Support Vector Classifier: RBF kernel')

# Compare decision boundries with polynomial kernel, degree = 3
plot_class_regions_for_classifier(SVC(kernel = 'poly', degree = 3)
                                 .fit(X_train, y_train), X_train,
                                 y_train, None, None,
                                 'Support Vector Classifier: Polynomial kernel, degree = 3')



#Support Vector Machine with RBF kernel: gamma parameter

from adspy_shared_utilities import plot_class_regions_for_classifier

X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state = 0)
fig, subaxes = plt.subplots(3, 1, figsize=(4, 11))

for this_gamma, subplot in zip([0.01, 1.0, 10.0], subaxes):
    clf = SVC(kernel = 'rbf', gamma=this_gamma).fit(X_train, y_train)
    title = 'Support Vector Classifier: \nRBF kernel, gamma = {:.2f}'.format(this_gamma)
    plot_class_regions_for_classifier_subplot(clf, X_train, y_train,
                                             None, None, title, subplot)
    plt.tight_layout()


#Support Vector Machine with RBF kernel: using both C and gamma parameter

from sklearn.svm import SVC
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state = 0)
fig, subaxes = plt.subplots(3, 4, figsize=(15, 10), dpi=50)

for this_gamma, this_axis in zip([0.01, 1, 5], subaxes):
    
    for this_C, subplot in zip([0.1, 1, 15, 250], this_axis):
        title = 'gamma = {:.2f}, C = {:.2f}'.format(this_gamma, this_C)
        clf = SVC(kernel = 'rbf', gamma = this_gamma,
                 C = this_C).fit(X_train, y_train)
        plot_class_regions_for_classifier_subplot(clf, X_train, y_train,
                                                 X_test, y_test, title,
                                                 subplot)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
'''

# Application of SVMs to a real dataset: unnormalized data

from sklearn.svm import SVC
#X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer,
#                                                   random_state = 0)

clf = SVC(C=10).fit(X_train, y_train)
print('fashion dataset (unnormalized features)')
print('Accuracy of RBF-kernel SVC on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of RBF-kernel SVC on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))
#Application of SVMs to a real dataset: normalized data with feature preprocessing using minmax scaling

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)




####################
# Learning  curve
####################
from sklearn.model_selection import learning_curve

# fashion  Learning Curve
# ----
train_sizes, train_scores, test_scores = learning_curve(DecisionTreeClassifier(random_state = 0, max_depth = 8), X_train_fashion, y_train_fashion, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)

print "train sizes", train_sizes
print "train scores", train_scores
print "test scores", test_scores

fig = plt.figure()
plt.title("fashion | Decision Tree Learning Curves (Max Depth = 8)")
plt.xlabel("Training examples")
plt.ylabel("Accuracy ( % Correctly Labeled) ")

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
#plt.plot(range(10))
fig.savefig('dt_fash_lc.png')

# Run Walk Learning Curve
# ----
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(DecisionTreeClassifier(random_state = 0, max_depth = 13), X_train_runwalk, y_train_runwalk, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)

print "train sizes", train_sizes
print "train scores", train_scores
print "test scores", test_scores

fig = plt.figure()
plt.title("runwalk | Decision Tree Learning Curves (Max Depth = 13)")
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

fig.savefig('dt_runwalk_lc.png')

####################
#Validation curve
####################

from sklearn.model_selection import validation_curve




####################
# VC  example
####################



param_range = np.logspace(-3, 3, 4)
train_scores, test_scores = validation_curve(SVC(), X, y,
                                            param_name='gamma',
                                            param_range=param_range, cv=3)

print(train_scores)
print(test_scores)

clf = SVC(C=10).fit(X_train_scaled, y_train)
print('fashion dataset (normalized with MinMax scaling)')
print('RBF-kernel SVC (with MinMax scaling) training set accuracy: {:.2f}'
     .format(clf.score(X_train_scaled, y_train)))
print('RBF-kernel SVC (with MinMax scaling) test set accuracy: {:.2f}'
     .format(clf.score(X_test_scaled, y_test)))

######################


param_range = np.linspace(1, 12, 12, dtype=int)
print param_range
#train_scores, test_scores = validation_curve(DecisionTreeClassifier(random_state = 0), X_train_fashion, y_train_fashion,
#                                            param_name='max_depth',
#                                            param_range=param_range, cv=5)

train_scores, test_scores = validation_curve(SVC(), X, y,
                                            param_name='gamma',
                                            param_range=param_range, cv=3)

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

plt.title('Decision Tree Model Complexity for Max Depth | fashion dataset')
plt.xlabel('Maximum Depth')
plt.ylabel('Accuracy ( % Correctly Labeled)')
plt.grid(True)
plt.ylim(0, 101)
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
fig.savefig('dt_fash_mc.png')

# runwalk model complexity
# ----------------
param_range = np.linspace(1, 20, 20, dtype=int)
print param_range
#train_scores, test_scores = validation_curve(DecisionTreeClassifier(random_state = 0), X_train_runwalk, y_train_runwalk,
#                                            param_name='max_depth',
#                                            param_range=param_range, cv=5)

train_scores, test_scores = validation_curve(SVC(), X_train_runwalk, y_train_runwalk,
                                            param_name='gamma',
                                            param_range=param_range, cv=3)

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

plt.title('Decision Tree Model Complexity for Max Depth | runwalk dataset')
plt.xlabel('Maximum Depth')
plt.ylabel('Accuracy ( % Correctly Labeled )')
plt.grid(True)
plt.ylim(80, 101)
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
fig.savefig('dt_runwalk_mc.png')


## Testset performance

clf2 = DecisionTreeClassifier(max_depth = 9, random_state = 0).fit(X_train_fashion, y_train_fashion)
print('fashion | Accuracy of Decision Tree classifier on training set max_depth = 9: {:.2f}'
     .format(clf2.score(X_train_fashion, y_train_fashion)))
print('fashion | Accuracy of Decision Tree classifier on test set max_depth = 9: {:.2f}'
     .format(clf2.score(X_test_fashion, y_test_fashion)))


clf = DecisionTreeClassifier(max_depth = 14, random_state = 0).fit(X_train_runwalk, y_train_runwalk)

print('runwalk | Accuracy of Decision Tree classifier on training set max_depth = 14: {:.2f}'
     .format(clf.score(X_train_runwalk, y_train_runwalk)))
print('runwalk | Accuracy of Decision Tree classifier on test set max_depth = 14: {:.2f}'
     .format(clf.score(X_test_runwalk, y_test_runwalk)))
