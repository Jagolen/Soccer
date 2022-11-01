import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.model_selection as ms
import pandas as pd

#Import the training data and define the output

data = pd.read_csv('train.csv')
output = data['Lead']
data.info()
#5 fittings will be tested: the output will always be Lead, the inputs will be:
# Number of words by males and females, Year of release, money made, all three,
# The same inputs as the other methods tested and all input data. These will be numbered 1 to 6 in order of apperance in the list and can be changed to test different feature importances
missc = np.zeros(6)
tests = [data[['Number words female', 'Number words male']], data[['Year']], data[['Gross']],\
    data[['Number words female', 'Number words male','Year','Gross']],\
        data.drop(['Lead', 'Total words', 'Number of words lead', 'Gross', 'Year'], axis = 1),data.drop(['Lead'], axis = 1)]

#10-fold crossvalidation is performed using the kfold method from sklearn
#The data is split into 10 parts, treat 9 of them as training data and 1 as test data
#The data is then cycled through untill all parts has been treated as test data

folds = 10
crossval = ms.KFold(n_splits=folds,random_state=2,shuffle=True)

#For each of the 6 different combinations of inputs selected, 10-fold validation is performed by calculating the mean of predicted
#outputs that are not the same as the output from the selected data for each of the 10 test datas, which is then divided by the number of
#folds to give an average estimation of the missclassification with the chosen inputs

for i in range(6):
    for train, test in crossval.split(tests[i]):

        #Training and testing inputs
        xtr, xte = tests[i].iloc[train],tests[i].iloc[test]

        #Training and testing outputs
        ytr, yte = output.iloc[train], output.iloc[test]

        #The model is implemented and fitted to the training data using a Limited-memory Broyden–Fletcher–Goldfarb–Shanno Algorithm (lbfgs)
        #with a maximum allowed number of iterations of 10000
        model = lm.LogisticRegression(max_iter = 10000, solver='lbfgs')
        model.fit(xtr, ytr)

        #The nodel predicts the output
        PredictedOutput = model.predict(xte)

        #The missclassification
        missc[i] += np.mean(PredictedOutput != yte)
missc /= folds

#The missclacification is then plotted based on the combinations of inputs.
print(missc) #The numerical values are also printed
plt.plot(range(1,7), missc)
plt.show()


#-----------------------BOXPLOT

misbox = []

crossval = ms.KFold(n_splits=folds,random_state=2,shuffle=True)
for train, test in crossval.split(tests[4]):

    #Training and testing inputs
    xtr, xte = tests[4].iloc[train],tests[4].iloc[test]

    #Training and testing outputs
    ytr, yte = output.iloc[train], output.iloc[test]

    #The model is implemented and fitted to the training data using a Limited-memory Broyden–Fletcher–Goldfarb–Shanno Algorithm (lbfgs)
    #with a maximum allowed number of iterations of 10000
    model = lm.LogisticRegression(max_iter = 10000, solver='lbfgs')
    model.fit(xtr, ytr)

    #The nodel predicts the output
    PredictedOutput = model.predict(xte)

    #The missclassification
    misbox.append(np.mean(PredictedOutput != yte))

plt.boxplot(misbox)
plt.show()