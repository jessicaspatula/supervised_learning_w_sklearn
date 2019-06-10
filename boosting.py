#########################
# Boosted Gradient Boosteds
#########################

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# fashion dataset

#fashion = pd.read_table('fashion_tenth.csv',sep=',')
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
# Learning  curve
####################
from sklearn.model_selection import learning_curve

# fashion  Learning Curve
# ----
train_sizes, train_scores, test_scores = learning_curve(GradientBoostingClassifier(random_state = 0), X_train_fashion, y_train_fashion, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)

print "train sizes", train_sizes
print "train scores", train_scores
print "test scores", test_scores

fig = plt.figure()
plt.title("fashion | Gradient Boosted Learning Curves (Max Depth = 8)")
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
fig.savefig('gb_fash_lc.png')

# Run Walk Learning Curve
# ----
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(GradientBoostingClassifier(random_state = 0), X_train_runwalk, y_train_runwalk, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)

print "train sizes", train_sizes
print "train scores", train_scores
print "test scores", test_scores

fig = plt.figure()
plt.title("runwalk | Gradient Boosted Learning Curves (Max Depth = 13)")
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

fig.savefig('gb_runwalk_lc.png')


####################
#Validation curve
####################

from sklearn.model_selection import validation_curve

#param_range = np.linspace(1, 12, 12, dtype=int)
param_range = np.linspace(0.1, 1 ,5 )
print param_range
train_scores, test_scores = validation_curve(GradientBoostingClassifier(), X_train_fashion, y_train_fashion,
                                            param_name='learning_rate',
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

plt.title('Gradient Boosted Model Complexity for Max Depth | fashion dataset')
plt.xlabel('Maximum Depth')
plt.ylabel('Accuracy ( % Correctly Labeled)')
plt.grid(True)
plt.ylim(0, 105)
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
fig.savefig('gb_fash_mc.png')

# runwalk model complexity
# ----------------
param_range = np.linspace(0.1,1,5)
print param_range
train_scores, test_scores = validation_curve(GradientBoostingClassifier(), X_train_runwalk, y_train_runwalk,
                                            param_name='learning_rate',
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

plt.title('Gradient Boosted Model Complexity for Max Depth | runwalk dataset')
plt.xlabel('Maximum Depth')
plt.ylabel('Accuracy ( % Correctly Labeled )')
plt.grid(True)
plt.ylim(80, 105)
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
fig.savefig('gb_runwalk_mc.png')



## Testset performance
clf2 = GradientBoostingClassifier( random_state = 0).fit(X_train_fashion, y_train_fashion)
print('fashion | Accuracy of Gradient Boosted DT classifier on training set : {:.2f}'
     .format(clf2.score(X_train_fashion, y_train_fashion)))
print('fashion | Accuracy of Gradient Boosted DT classifier on test set : {:.2f}'
     .format(clf2.score(X_test_fashion, y_test_fashion)))

clf = GradientBoostingClassifier( random_state = 0).fit(X_train_runwalk, y_train_runwalk)
print('runwalk | Accuracy of Gradient Boosted DT classifier on training set : {:.2f}'
     .format(clf.score(X_train_runwalk, y_train_runwalk)))
print('runwalk | Accuracy of Gradient Boosted DT classifier on test set : {:.2f}'
     .format(clf.score(X_test_runwalk, y_test_runwalk)))


