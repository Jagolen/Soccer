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

#from IPython.display import set_matplotlib_formats
#set_matplotlib_formats('png')
#from IPython.core.pylabtools import figsize
#figsize(10, 6) # Width and hight
#plt.style.use('seaborn-white')

path = 'train.csv'
all_data = pd.read_csv(path, na_values='?', dtype={'ID': str}).dropna().reset_index()
y_name = "Lead"

np.random.seed(1)

n_fold = 10
k_neighbors = 5

# X = all_data.drop(columns=[y_name])
X = all_data.drop(columns=['Lead', 'Total words', 'Number of words lead', 'Gross', 'Year'])
y = all_data[y_name]

models = [skl_lm.LogisticRegression(solver="liblinear"),
          skl_da.LinearDiscriminantAnalysis(),
          skl_da.QuadraticDiscriminantAnalysis(),
          skl_nb.KNeighborsClassifier(n_neighbors=k_neighbors),
          RandomForestClassifier(criterion='gini', bootstrap=True, n_estimators=70, max_depth=5)]
models_string = ["Logistic Regression", "LDA", "QDA", f"kNN, k={k_neighbors}", "Random Forest"]

misclassifications = np.zeros((n_fold, len(models)))
misclassifications_train = np.zeros((n_fold, len(models)))
cv = skl_ms.KFold(n_splits=n_fold, random_state=1, shuffle=True)

for i, (train_index, val_index) in enumerate(cv.split(X)):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    for m, model in enumerate(models):
        model.fit(X_train, y_train)
        prediction = model.predict(X_val)

        prediction_train = model.predict(X_train)
        misclassifications_train[i, m] = np.mean(prediction_train != y_train)

        misclassifications[i, m] = np.mean(prediction != y_val)

plt.figure(1)
plt.boxplot(misclassifications)
plt.title("Cross validation error for different methods")
plt.xticks(np.arange(len(models_string)) + 1, models_string)
plt.ylabel("validation error")
plt.ylim([0.075, 0.3])

plt.figure(2)
plt.boxplot(misclassifications_train)
plt.title("Cross validation training error for different methods")
plt.xticks(np.arange(len(models_string)) + 1, models_string)
plt.ylabel("validation error")
plt.ylim([0.075, 0.3])
# plt.show()


misclassifications = np.zeros((n_fold, len(models)))
misclassifications_train = np.zeros((n_fold, len(models)))
cv = skl_ms.KFold(n_splits=n_fold, random_state=1, shuffle=True)

for i, (train_index, val_index) in enumerate(cv.split(X)):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    scaler = skl_pre.MinMaxScaler().fit(X_train)

    for m, model in enumerate(models):
        model.fit(scaler.transform(X_train), y_train)
        prediction = model.predict(scaler.transform(X_val))

        prediction_train = model.predict(scaler.transform(X_train))
        misclassifications_train[i, m] = np.mean(prediction_train != y_train)


        # model.fit(X_train, y_train)
        # prediction = model.predict(X_val)
        misclassifications[i, m] = np.mean(prediction != y_val)

plt.figure(3)
plt.boxplot(misclassifications)
plt.title("Cross validation error for different methods, with a scaler")
plt.xticks(np.arange(len(models_string)) + 1, models_string)
plt.ylabel("validation error")
plt.ylim([0.075, 0.3])

plt.figure(4)
plt.boxplot(misclassifications_train)
plt.title("Cross validation training error for different methods, with a scaler")
plt.xticks(np.arange(len(models_string)) + 1, models_string)
plt.ylabel("validation error")
plt.ylim([0.075, 0.3])
plt.show()
