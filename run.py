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
ANALYSIS = 'PLSR_Bruno_cv_250'
save_results = True

if FOLDER_BASE[-1] != '/':
    FOLDER_BASE += '/'

if not os.path.exists(f"{FOLDER_BASE}{ANALYSIS}"):
    os.mkdir(f"{FOLDER_BASE}{ANALYSIS}")
    FOLDER = f"{FOLDER_BASE}{ANALYSIS}"
else:
    FOLDER = f"{FOLDER_BASE}{ANALYSIS}"

#df = pd.read_csv('/home/dsilva/testes_ml/dataset/carol_correct/ep-eb_calibration.csv', sep=';', decimal=',')
#df_val = pd.read_csv('/home/dsilva/testes_ml/dataset/carol_correct/ep_for_prediction.csv', sep=';', decimal=',')
df = pd.read_csv('/home/dsilva/testes_ml/dataset/bruno/bruno_250_divisao_bruno_sgr.csv', sep=';', decimal=',')
#df_val = pd.read_csv('/home/dsilva/testes_ml/dataset/bruno/bruno_100_divisao_bruno_sgr.csv', sep=';', decimal=',')



# df = make_average(df, 2, 2)
# df_val = make_average(df_val, 2, 2)

print('Fazendo transformações... ')
transformations = make_transformations([df], ['msc'], 2)
print('Ok')
# TRANSFORMAÇÕES

results = np.zeros((len(transformations), 7))
results[:] = np.nan

df_results = pd.DataFrame(results)

"""
print('Começando a criar os modelos...')
for pos, transformation in enumerate(transformations):
    rf = PLSR(transformation[0], components=5, cross_validation_type='loo', split_for_validation='all')
    #rf.search_hyperparameters(n_processors=-1, verbose=1, estimators=[100, 510, 100], min_samples_split=[2, 10, 100, 400], min_samples_leaf=[1,2, 3])
    rf.search_hyperparameters(n_processors=-1, verbose=1, components=[1, 21, 1])
    rf.calibrate()
    rf.cross_validate()

    if save_results == True:
        save_results_of_model(rf, path=FOLDER, name=transformation[1], plots=True, out_table=True, coefficients_of_model='plsr')


    try:
        df_results.iloc[pos, 0] = transformation[1]
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

    print(transformation[1])


df_results.columns = ['TRANSFORMATION', 'R2_CAL', 'RMSE_CAL', 'R2_CV', 'RMSE_CV', 'R2_PRED', 'RMSE_PRED']
df_results.to_csv(f"{FOLDER}/results_{ANALYSIS}.csv", sep=';', decimal=',', index=False)"""