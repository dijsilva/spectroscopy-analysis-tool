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
ANALYSIS = 'BrunoCarol_EB'
save_results = True

if FOLDER_BASE[-1] != '/':
    FOLDER_BASE += '/'

if not os.path.exists(f"{FOLDER_BASE}{ANALYSIS}"):
    os.mkdir(f"{FOLDER_BASE}{ANALYSIS}")
    FOLDER = f"{FOLDER_BASE}{ANALYSIS}"
else:
    FOLDER = f"{FOLDER_BASE}{ANALYSIS}"

cal = pd.read_csv('/home/dsilva/testes_ml/dataset/bruno_carol/calibration.csv', sep=';', decimal=',')
val = pd.read_csv('/home/dsilva/testes_ml/dataset/carol_corrigido/eb.csv', sep=';', decimal=',')

#df = make_average(df, 2, 2)
val = make_average(val, 2, 2)

print('Fazendo transformações... ')
transformations = make_transformations([cal, val], ['all'], 2)
print(' Ok')

results = np.zeros((len(transformations), 9))
results[:] = np.nan
df_results = pd.DataFrame(results)

print('Criando os modelos...')
for pos, transformation in enumerate(transformations):
    # rf = PLSR(transformation[0], components=1, cross_validation_type=10, dataset_validation=transformation[1])
    # rf.test_many_components(components=[1,21], target='pred')
    # rf.search_hyperparameters(n_processors=-1, verbose=1, components=[1, 21, 1])

    model = RandomForest(transformation[0], estimators=100, cross_validation_type=10, dataset_validation=transformation[1])
    #model.search_hyperparameters(n_processors=-1, verbose=1, estimators=[50, 400, 100])

    # model = SVMRegression(transformation[0], type_of_kernel='rbf', cross_validation_type='loo', dataset_validation=transformation[1])
    # model.search_hyperparameters(n_processors=3, epsilon=[ 0.1, 2.0, 0.2 ], verbose=1, kernel = ['rbf'], gamma=[100])

    # model = PCR(transformation[0], components=11, cross_validation_type='loo', dataset_validation=transformation[1])
    # model.search_hyperparameters(components=[1, 21, 1])

    model.create_model()

    if save_results == True:
        save_results_of_model(model, path=FOLDER, name=transformation[2], plots=True, out_table=True, out_performance=True, coefficients_of_model='random_forest')


    try:
        df_results.iloc[pos, 0] = transformation[2]
        df_results.iloc[pos, 1] = model.metrics['calibration']['R2']
        df_results.iloc[pos, 2] = model.metrics['calibration']['RMSE']
    except:
        raise ValueError('a error occurred with data of calibration')


    try:
        df_results.iloc[pos, 3] = model.metrics['cross_validation']['R2']
        df_results.iloc[pos, 4] = model.metrics['cross_validation']['RMSE']
        df_results.iloc[pos, 5] = model.metrics['cross_validation']['bias']
    except:
        pass


    try:
        df_results.iloc[pos, 6] = model.metrics['validation']['R2']
        df_results.iloc[pos, 7] = model.metrics['validation']['RMSE']
        df_results.iloc[pos, 8] = model.metrics['validation']['bias']
    except:
        pass

    print(f"{transformation[2]} - Ok")


df_results.columns = ['TRANSFORMATION', 'R2_CAL', 'RMSE_CAL', 'R2_CV', 'RMSE_CV', 'BIAS_CV', 'R2_PRED', 'RMSE_PRED', 'BIAS_PRED']
df_results.to_csv(f"{FOLDER}/results_{ANALYSIS}.csv", sep=';', decimal=',', index=False)