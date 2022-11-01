import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn.model_selection as skl_ms
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

#
# Using Classification tree (CT) and Random Forest (RF) methods to classify the gender of the two main actors in movies 
# We have a training set which consists of 1039 films and we will later be given a test set of 387 films
# all features in the data are qualitative variables, except the output 'Lead' (either Male or Female)
#

# Our data 
auto_tr = pd.read_csv('train.csv')

# Creating predictable random numbers
np.random.seed(1)

# Feature importance for both methods
features = auto_tr.iloc[:,0:14]            # The colums in the data
features = auto_tr.drop(columns=['Lead'])  # Removing the traget column
output = auto_tr.iloc[:,-1]                # The target column, 'Lead'

model = tree.DecisionTreeClassifier()      # tree.DecisionTreeClassifier() for CT and RandomForestClassifier() for RF
model.fit(features,output)                 # Building the mddel from features and output

# Using feature_importances_ function from sklearn and then creating an array with data and their corresponding indices (feature)
feature_importance = pd.Series(model.feature_importances_, index = features.columns)  
feature_importance.nlargest(13).plot(kind='barh')   # Plotting all features according to their importance


#################################################                             #################################################
################################################# Classification tree method  #################################################
#################################################                             #################################################

# After finding the optimal number of features with GridSearchCV(), 7 most important features have been selected as inputs
X_train_CT = auto_tr.drop(columns=['Lead','Year','Total words','Gross','Number words male', 'Number of words lead', 'Mean Age Female'])
Y_train_CT = auto_tr['Lead']

# Hyperparameters tuning using the function GridSearchCV()
# The hyperparameters that will be tuned

max_depth = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
max_leaf_nodes = [10,15,20,25,30,35,40,45,50]
criterion = ['gini', 'entropy']
splitter = ['best', 'random']
max_features = [1,2,3,4,5,6,7,8,9,10,11,12,13]
parameters = {'max_leaf_nodes': max_leaf_nodes,
                'max_depth': max_depth,
                'criterion': criterion,
                'splitter':splitter,
                'max_features':max_features}
CT = tree.DecisionTreeClassifier() 
grid_search = GridSearchCV(estimator = CT, param_grid = parameters, cv = 5)  # Seaching the optimal values with different cv (folds)
grid_search.fit(X_train_CT, Y_train_CT)  # Here X_train_CT consisted of alla 13 inputs
print(grid_search.best_params_)

# The final CT model after hyperparameters tuning.
model_CT = tree.DecisionTreeClassifier(criterion = 'entropy', splitter = 'best', max_leaf_nodes = 25 ,max_depth = 9)
model_CT.fit(X_train_CT, Y_train_CT)
prediction_CT = model_CT.predict(X_train_CT)  # Prediction on inputs

# Cross validation
folds = 10      # k-fold cross validation
depth = np.arange(1,50)
misclassification_CT = np.zeros((folds, len([model_CT])))
error_CT_depth = np.zeros((len(depth)))

# Splitting data into train and validation sets with k folds and with shuffle
cv = skl_ms.KFold(n_splits = folds, random_state = 1, shuffle = True)
for  i, (train_index, val_index) in enumerate(cv.split(X_train_CT)):
    X_tr_CT, X_val_CT = X_train_CT.iloc[train_index], X_train_CT.iloc[val_index]
    Y_tr_CT, Y_val_CT = Y_train_CT.iloc[train_index], Y_train_CT.iloc[val_index]

     # Evaluating the performance on the validation data by CT
    model_CT.fit(X_tr_CT, Y_tr_CT)
    pred_CT = model_CT.predict(X_val_CT)
    misclassification_CT[i] = np.mean(pred_CT != Y_val_CT) # The mean of the error 
    
    # Checking how the valdiation error depends on the depth (After selecting the important inputs)
    for j, dep in enumerate(depth):
        CT_depth = RandomForestClassifier(max_depth = dep) 
        CT_depth.fit(X_tr_CT, Y_tr_CT)
        pred_CT_depth = CT_depth.predict(X_val_CT)
        error_CT_depth[j] += np.mean(pred_CT_depth != Y_val_CT)

error_CT_depth = error_CT_depth/folds
misclassification_CT = misclassification_CT/folds

# Return the result
print('-------------------------------------------------------')
print('Train error rate with Classification tree: %.3f' % np.mean(prediction_CT != Y_train_CT))
print("Confusion matrix for Classification tree:\n")
print(pd.crosstab(prediction_CT, Y_train_CT), '\n')
print('-------------------------------------------------------')

#################################################                             #################################################
#################################################     Random forest method    #################################################
#################################################                             #################################################

# After finding the optimal number of features with GridSearchCV(), 9 most important features have been selected as inputs
X_train_RF = auto_tr.drop(columns=['Lead','Total words', 'Number of words lead', 'Gross', 'Year'])
Y_train_RF = auto_tr['Lead']

# The hyperparameters that will be tuned
max_depth = [3,4,5,6,7,8,9,10,11]
n_estimators = [40,45,50,55,60,65,70,75,80,85,90,95,100]
criterion = ['gini', 'entropy']
max_features = [1,2,3,4,5,6,7,8,9,10,11,12,13]
bootstrap = ['True', 'False']

parameters_RF = {'max_depth': max_depth,
                'criterion': criterion,
                'max_features':max_features,
                'n_estimators':n_estimators,
                'bootstrap':bootstrap}

RF = RandomForestClassifier() 
grid_search_RF = GridSearchCV(estimator = RF, param_grid = parameters_RF, cv = 3)
grid_search_RF.fit(X_train_RF, Y_train_RF)  # Here X_train_CT consisted of alla 13 inputs
print(grid_search_RF.best_params_)

# The final RF model after hyperparameters tuning
model_RF = RandomForestClassifier(criterion='gini', bootstrap = True, n_estimators = 70, max_depth = 5)
model_RF.fit(X_train_RF, Y_train_RF)
prediction_RF = model_RF.predict(X_train_RF)  # Prediction on inputs

# Cross validation
n_est = np.arange(1,100)
depth = np.arange(1,50)
misclassification_RF = np.zeros((folds, len([model_RF])))
error_RF_n_est = np.zeros((len(n_est)))
error_RF_depth = np.zeros((len(depth)))

for  i, (train_index, val_index) in enumerate(cv.split(X_train_RF)):
    X_tr_RF, X_val_RF = X_train_RF.iloc[train_index], X_train_RF.iloc[val_index]
    Y_tr_RF, Y_val_RF = Y_train_RF.iloc[train_index], Y_train_RF.iloc[val_index]

    # Evaluating the performance on the validation data by CT.
    model_RF.fit(X_tr_RF, Y_tr_RF)
    pred_RF = model_RF.predict(X_val_RF)
    misclassification_RF[i] = np.mean(pred_RF != Y_val_RF)  # The mean of the error

    # Checking how the valdiation error depends on the number of trees (After selecting the important inputs) 
    for j, n in enumerate(n_est):
        RF_n_est = RandomForestClassifier(n_estimators = n)
        RF_n_est.fit(X_tr_RF, Y_tr_RF)
        pred_RF_n_est = RF_n_est.predict(X_val_RF)
        error_RF_n_est[j] += np.mean(pred_RF_n_est != Y_val_RF)

    # Checking how the valdiation error depends on the depth (After selecting the important inputs)
    for j, d in enumerate(depth):
        RF_depth = RandomForestClassifier(max_depth = d) 
        RF_depth.fit(X_tr_RF, Y_tr_RF)
        pred_RF_depth = RF_depth.predict(X_val_RF)
        error_RF_depth[j] += np.mean(pred_RF_depth != Y_val_RF)

error_RF_n_est = error_RF_n_est/folds
error_RF_depth = error_RF_depth/folds
misclassification_RF = misclassification_RF/folds

# Return the result
print('Train error rate with Random Forests: %.3f' % np.mean(prediction_RF != Y_train_RF))
print("Confusion matrix for Random Forests:\n")
print(pd.crosstab(prediction_RF, Y_train_RF), '\n')
print('-------------------------------------------------------')

#------------------------------------------------------------------------------------------------------------------------------
# *****************************************************************************************************************************
# *****************************************************************************************************************************
# *****************************************************************************************************************************
#------------------------------------------------------------------------------------------------------------------------------

#
# The figures
# See the titles of the figures to understand what they represent
# 

plot1 = plt.figure(1)
plt.plot(depth, error_CT_depth)
plt.title('Cross validation error for max depth, CT method')
plt.xlabel('max_depth')
plt.ylabel('validation error')
plot1.show()

plot2 = plt.figure(2)
plt.plot(depth, error_RF_depth)
plt.title('Cross validation error for max depth, RF method')
plt.xlabel('max_depth')
plt.ylabel('validation error')
plot2.show()

plot3 = plt.figure(3)
plt.plot(n_est, error_RF_n_est)
plt.title('Cross validation error for different number of trees, RF method')
plt.xlabel('n_estimators')
plt.ylabel('validation error')
plot3.show()

plot4 = plt.figure(4)
plt.boxplot(misclassification_CT)
plt.title('Cross validation error for Classification tree')
plt.xlabel('Classification tree')
plt.ylabel('validation error')
plot4.show()

plot5 = plt.figure(5)
plt.boxplot(misclassification_RF)
plt.title('Cross validation error for Random Forest method')
plt.xlabel('Random Forest')
plt.ylabel('validation error')
plot5.show()
plt.show()
