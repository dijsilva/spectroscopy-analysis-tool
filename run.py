import os
import sys
sys.path.insert(0, '/home/dsilva/projects/git/spectroscopy-pca-lda/')


from algorithms.regression import PCR, RandomForest, PLSR, SVMRegression
from algorithms.classification import PCA_LDA

from transformations import make_transformations

from utils import make_average
from utils.handle_results import save_results_of_model

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#VARIABLES
FOLDER_BASE = '/home/dsilva/testes_ml/models'
ANALYSIS = 'SVM_Regression_eb'
save_results = True

if FOLDER_BASE[-1] != '/':
    FOLDER_BASE += '/'

if not os.path.exists(f"{FOLDER_BASE}{ANALYSIS}"):
    os.mkdir(f"{FOLDER_BASE}{ANALYSIS}")
    FOLDER = f"{FOLDER_BASE}{ANALYSIS}"
else:
    FOLDER = f"{FOLDER_BASE}{ANALYSIS}"

df = pd.read_csv('/home/dsilva/testes_ml/dataset/carol_correct/ep-eb_calibration.csv', sep=';', decimal=',')
df_val = pd.read_csv('/home/dsilva/testes_ml/dataset/carol_correct/eb_for_prediction.csv', sep=';', decimal=',')


# df = make_average(df, 2, 2)
# df_val = make_average(df_val, 2, 2)

transformeds = make_transformations([df_val], ['all'], 2)

# TRANSFORMAÇÕES



"""

results = np.zeros((len(transformations), 7))
results[:] = np.nan

df_results = pd.DataFrame(results)

for pos, transformation in enumerate(transformations):
    rf = SVMRegression(transformation[0], cross_validation_type='loo', dataset_validation=transformation[1])
    #rf.search_hyperparameters(n_processors=-1, verbose=1, estimators=[100, 510, 100], min_samples_split=[2, 10, 100, 400], min_samples_leaf=[1,2, 3])
    rf.search_hyperparameters(n_processors=-1, verbose=1, kernel=['rbf', 'linear'], gamma=[50, 100, 200])
    rf.create_model()

    if save_results == True:
        save_results_of_model(rf, path=FOLDER, name=transformation[2], plots=True, out_table=True, coefficients_of_model='svr')


    try:
        df_results.iloc[pos, 0] = transformation[2]
        df_results.iloc[pos, 1] = rf.metrics['calibration']['R2']
        df_results.iloc[pos, 2] = rf.metrics['calibration']['RMSE']
    except:
        pass


    try:
        df_results.iloc[pos, 3] = rf.metrics['cross_validation']['R2']
        df_results.iloc[pos, 4] = rf.metrics['cross_validation']['RMSE']
    except:
        pass


    try:
        df_results.iloc[pos, 5] = rf.metrics['validation']['R2']
        df_results.iloc[pos, 6] = rf.metrics['validation']['RMSE']
    except:
        pass

    print(transformation[2])


df_results.columns = ['TRANSFORMATION', 'R2_CAL', 'RMSE_CAL', 'R2_CV', 'RMSE_CV', 'R2_PRED', 'RMSE_PRED']
df_results.to_csv(f"{FOLDER}/results.csv", sep=';', decimal=',', index=False)
"""