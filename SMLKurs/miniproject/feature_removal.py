import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms


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


path = 'train.csv'
all_data = pd.read_csv(path, na_values='?', dtype={'ID': str}).dropna().reset_index()

np.random.seed(1)
n_fold = 10

X = all_data.drop(columns=['Lead', 'index'])
y = all_data['Lead']

X_, X_test, y_, y_test = skl_ms.train_test_split(X, y, test_size=0.05, random_state=1)

# drop_from_all = ["Total words", "Gross", "Year", "Mean Age Female"]
# drop_from_all = ["Number of words lead", "Mean Age Male", "Age Co-Lead"]
drop_from_all = ["Gross", "Age Co-Lead", "Number of words lead"]
feature_list = [#"Difference in words lead and co-lead",
                
                #"Total words",
                #"Number of words lead",
                #"Number words male",
                #"Number words female",
                #"Gross",
                #"Age Lead",
                #"Age Co-Lead",
                #"Number of female actors",
                #"Number of male actors",
                #"Mean Age Female",
                #"Mean Age Male",
                #"Year"
                ]
features_to_test = [drop_from_all + [feature] for feature in feature_list if feature not in drop_from_all]
set_to_test = features_to_test

#model = skl_da.QuadraticDiscriminantAnalysis()

model = skl_lm.LogisticRegression(max_iter = 10000, solver='lbfgs')

misclassifications = np.zeros((n_fold, len(set_to_test) + 2))
misclassifications_train = np.zeros((n_fold, len(set_to_test) + 2))
cv = skl_ms.KFold(n_splits=n_fold, random_state=1, shuffle=True)

for i, (train_index, val_index) in enumerate(cv.split(X_)):
    X_train_, X_val_ = X.iloc[train_index], X_.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y_.iloc[val_index]

    a, b = fit_and_predict(model, X_train_, X_val_, y_train, y_val)
    misclassifications_train[i, -1] = a
    misclassifications[i, -1] = b

    a, b = fit_and_predict(model, X_train_.drop(columns=drop_from_all), X_val_.drop(columns=drop_from_all), y_train, y_val)
    misclassifications_train[i, -2] = a
    misclassifications[i, -2] = b

    for m, drop_features in enumerate(set_to_test):
        a, b = fit_and_predict(model, X_train_.drop(columns=drop_features), X_val_.drop(columns=drop_features), y_train, y_val)
        misclassifications_train[i, m] = a
        misclassifications[i, m] = b

# xlabeling = [str(x) for x in range(len(set_to_test))] + ["Current"] + ["All"]
xlabeling = [feature[0:min(len(feature), 14)] for feature in feature_list if feature not in drop_from_all] + ["Optimized"] + ["All features"]

plt.figure(1)
plt.boxplot(misclassifications)
plt.title("Cross validation error for different number of features")
plt.xticks(np.arange(len(set_to_test) + 2) + 1, xlabeling)
plt.ylabel("validation error")
# plt.ylim([0.075, 0.3])

# plt.figure(2)
# plt.boxplot(misclassifications_train)
# plt.title("Cross validation training error for different number of features")
# # plt.xticks(np.arange(len(models_string)) + 1, models_string)
# plt.ylabel("validation error")
# # plt.ylim([0.075, 0.3])
plt.show()
