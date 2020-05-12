from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

from algorithms.regression import PCR, RandomForest, PLSR
from algorithms.classification import PCA_LDA
from transformations import snv, sg, msc, plus_sg, area_norm
import pandas as pd
from utils import make_average
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('/home/dsilva/teste_pcr/foss_para_arff_calibracao.csv', sep=';', decimal=',')
df_val = pd.read_csv('/home/dsilva/teste_pcr/foss_para_arff_validacao.csv', sep=';', decimal=',')
df_all = pd.read_csv('/home/dsilva/teste_pcr/foss_all_samples.csv', sep=';', decimal=',')

rf = RandomForest(df, 100, 2, dataset_validation=df_val)
#rf.create_model()

"""
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10, 20]
min_samples_leaf = [1, 2, 4, 8]
bootstrap = [True]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
rf = RandomForestRegressor()

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 10, verbose=2, random_state=1, n_jobs = -1)

rf_random.fit(df.iloc[:, 2:], df.iloc[:, 1])
"""

# {'n_estimators': 1400, 'min_samples_split': 20, 'min_samples_leaf': 8, 'max_features': 'sqrt', 'max_depth': None, 'bootstrap': True}

# param_grid = {
#     'bootstrap': [True],
#     'max_depth': [80],
#     'max_features': [2, 3, 4],
#     'min_samples_leaf': [5, 7, 8, 9, 10],
#     'min_samples_split': [18, 19, 20, 21, 22],
#     'n_estimators': [1300, 1400, 1500, 1600, 1700]
# }

# rf = RandomForestRegressor()
# grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
#                           cv = 10, n_jobs = -1, verbose = 2)

# grid_search.fit(df.iloc[:, 2:], df.iloc[:, 1])

# {'bootstrap': True, 'max_depth': 80, 'max_features': 2, 'min_samples_leaf': 8, 'min_samples_split': 21, 'n_estimators': 1600}