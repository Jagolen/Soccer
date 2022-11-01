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

path = 'train.csv'
all_data = pd.read_csv(path, na_values='?', dtype={'ID': str}).dropna().reset_index()
avg = []
yr = []
for y in all_data['year']:
    if y not in yr:
        yr.append(year)
