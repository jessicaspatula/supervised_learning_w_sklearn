import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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

target_names_fashion = ['T-shirt/top' ,'Trouser' ,'Pullover' ,'Dress' ,'Coat' ,'Sandal' ,'Shirt' ,'Sneaker' ,'Bag' ,'Ankle boot']
feature_names_fashion = list(fashion.columns.values)[:-1]
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

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train_scaled, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train_scaled, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test_scaled, y_test)))

example_fruit = [[5.5, 2.2, 10, 0.70]]
example_fruit_scaled = scaler.transform(example_fruit)
print('Predicted fruit type for ', example_fruit, ' is ', 
          target_names_fruits[knn.predict(example_fruit_scaled)[0]-1])
'''

####################
# Cross-validation
####################
'''
# Example based on k-NN classifier with fruit dataset (2 features)

from sklearn.model_selection import cross_val_score

clf = KNeighborsClassifier(n_neighbors = 5)
X = X_fruits_2d.as_matrix()
y = y_fruits_2d.as_matrix()
cv_scores = cross_val_score(clf, X, y)

print('Cross-validation scores (3-fold):', cv_scores)
print('Mean cross-validation score (3-fold): {:.3f}'
     .format(np.mean(cv_scores)))

'''
####################
#Decision Trees
####################

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from adspy_shared_utilities import plot_decision_tree
from sklearn.model_selection import train_test_split
#Visualizing decision trees

'''
plot_decision_tree(clf, feature_names_fashion, target_names_fashion)
#Pre-pruned version (max_depth = 3)
plot_decision_tree(clf2, feature_names_fashion, target_names_fashion)

#Feature importance
from adspy_shared_utilities import plot_feature_importances
plt.figure(figsize=(10,4), dpi=80)
plot_feature_importances(clf, iris.feature_names)
plt.show()

print('Feature importances: {}'.format(clf.feature_importances_))

from sklearn.tree import DecisionTreeClassifier
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot

#X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state = 0)
fig, subaxes = plt.subplots(6, 1, figsize=(6, 32))

pair_list = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]
tree_max_depth = 4

for pair, axis in zip(pair_list, subaxes):
    X = X_train[:, pair]
    y = y_train
    
    clf = DecisionTreeClassifier(max_depth=tree_max_depth).fit(X, y)
    title = 'Decision Tree, max_depth = {:d}'.format(tree_max_depth)
    plot_class_regions_for_classifier_subplot(clf, X, y, None,
                                             None, title, axis,
                                             iris.target_names)
    
    axis.set_xlabel(iris.feature_names[pair[0]])
    axis.set_ylabel(iris.feature_names[pair[1]])
    
plt.tight_layout()
plt.show()
'''
#Decision Trees on a real-world dataset
from adspy_shared_utilities import plot_decision_tree
from adspy_shared_utilities import plot_feature_importances


'''
plt.figure(figsize=(10,6),dpi=80)
plot_feature_importances(clf, feature_names_fashion)
plt.tight_layout()
plt.show()
'''

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

param_range = np.linspace(1, 12, 12, dtype=int)
print param_range
train_scores, test_scores = validation_curve(DecisionTreeClassifier(random_state = 0), X_train_fashion, y_train_fashion,
                                            param_name='max_depth',
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
train_scores, test_scores = validation_curve(DecisionTreeClassifier(random_state = 0), X_train_runwalk, y_train_runwalk,
                                            param_name='max_depth',
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

