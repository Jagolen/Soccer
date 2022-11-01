import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter as pc
import matplotlib

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms

#from IPython.display import set_matplotlib_formats
#set_matplotlib_formats('png')
from IPython.core.pylabtools import figsize
figsize(10, 6) # Width and hight
#plt.style.use('seaborn-white')


def fit_and_predict(model, X_train_, X_val_, y_train, y_val, scaler=None):
    """Returns misclassification % during training and validating, for given data"""
    if scaler is not None:
        scaler.fit(X_train_)
        X_train = scaler.transform(X_train_)
        X_val = scaler.transform(X_val_)
    else:
        X_train = X_train_
        X_val = X_val_

    model.fit(X_train, y_train)

    prediction = model.predict(X_val)
    prediction_train = model.predict(X_train)

    misclassifications_training = np.mean(prediction_train != y_train)
    misclassifications_validating = np.mean(prediction != y_val)

    return misclassifications_training, misclassifications_validating


start = pc()

path = 'train.csv'
all_data = pd.read_csv(path, na_values='?', dtype={'ID': str}).dropna().reset_index()
y_name = "Lead"

np.random.seed(1)

n_fold = 10
k_neighbors = 25

# X = all_data.drop(columns=[y_name])
X = all_data.drop(columns=['Lead', 'index'])
y = all_data[y_name]

drop_features = [["Gross", "Age Co-Lead", "Number of words lead"],
                 ["Age Co-Lead"],
                 ["Number of words lead", "Mean Age Male", "Age Co-Lead"],
                 ["Number of words lead", "Total words", "Mean Age Male", "Age Co-Lead", "Number of female actors",
                  "Age Lead", "Number of male actors", "Year"],
                 ["Year", "Mean Age Female", "Age Lead", "Gross", "Number of words lead", "Number words male",
                  "Mean Age Male", "Total words", "Number words male"],
                 ['Year', 'Total words', 'Gross', 'Number words male', 'Number of words lead', 'Mean Age Female'],
                 ["Age Co-Lead"]]


models = [skl_lm.LogisticRegression(solver="lbfgs", max_iter=10000),
          skl_da.LinearDiscriminantAnalysis(),
          skl_da.QuadraticDiscriminantAnalysis(),
          skl_nb.KNeighborsClassifier(n_neighbors=k_neighbors),
          RandomForestClassifier(criterion='gini', bootstrap=True, n_estimators=40, max_depth=7),
          tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_leaf_nodes=25, max_depth=10),
          AdaBoostClassifier()]
models_string = ["Logistic Regression", "LDA", "QDA", f"kNN, k={k_neighbors}", "Random Forest", "Classification tree", "Adaboost"]

misclassifications = np.zeros((n_fold, len(models)))
misclassifications_train = np.zeros((n_fold, len(models)))
cv = skl_ms.KFold(n_splits=n_fold, random_state=1, shuffle=True)

for i, (train_index, val_index) in enumerate(cv.split(X)):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    for m, model in enumerate(models):
        a, b = fit_and_predict(model, X_train.drop(columns=drop_features[m]), X_val.drop(columns=drop_features[m]), y_train, y_val)
        misclassifications_train[i, m] = a
        misclassifications[i, m] = b

lim_plot = [0.055, 0.3]

plt.figure(1)
plt.boxplot(misclassifications)
plt.title("Cross validation error for different methods, features optimized for each model")
plt.xticks(np.arange(len(models_string)) + 1, models_string)
plt.ylabel("validation error")
plt.ylim(lim_plot)

plt.figure(2)
plt.boxplot(misclassifications_train)
plt.title("Cross validation training error for different methods, features optimized for each model")
plt.xticks(np.arange(len(models_string)) + 1, models_string)
plt.ylabel("validation error")
plt.ylim(lim_plot)
# plt.show()


# misclassifications = np.zeros((n_fold, len(models)))
# misclassifications_train = np.zeros((n_fold, len(models)))
# cv = skl_ms.KFold(n_splits=n_fold, random_state=1, shuffle=True)
#
# for i, (train_index, val_index) in enumerate(cv.split(X)):
#     X_train, X_val = X.iloc[train_index], X.iloc[val_index]
#     y_train, y_val = y.iloc[train_index], y.iloc[val_index]
#
#     scaler = skl_pre.MinMaxScaler()
#
#     for m, model in enumerate(models):
#         a, b = fit_and_predict(model, X_train, X_val, y_train, y_val, scaler)
#         misclassifications_train[i, m] = a
#         misclassifications[i, m] = b

stop = pc()
print(f"Time: {stop-start:.3f}")

# plt.figure(3)
# plt.boxplot(misclassifications)
# plt.title("Cross validation error for different methods, with a scaler")
# plt.xticks(np.arange(len(models_string)) + 1, models_string)
# plt.ylabel("validation error")
# plt.ylim(lim_plot)
#
# plt.figure(4)
# plt.boxplot(misclassifications_train)
# plt.title("Cross validation training error for different methods, with a scaler")
# plt.xticks(np.arange(len(models_string)) + 1, models_string)
# plt.ylabel("validation error")
# plt.ylim(lim_plot)

plt.show()
