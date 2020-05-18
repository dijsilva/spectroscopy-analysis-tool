import os
import sys
sys.path.insert(0, '/home/dsilva/projects/git/spectroscopy-pca-lda/')


from algorithms.regression import PCR, RandomForest, PLSR
from algorithms.classification import PCA_LDA
from transformations import snv, sg, msc, plus_sg, area_norm

from utils import make_average
from utils.handle_results import save_results_of_model

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#VARIABLES
FOLDER_BASE = '/home/dsilva/testes_ml/models'
ANALYSIS = 'RandomForest_EP-EB_EP'
save_results = False

if FOLDER_BASE[-1] != '/':
    FOLDER_BASE += '/'

if not os.path.exists(f"{FOLDER_BASE}{ANALYSIS}"):
    os.mkdir(f"{FOLDER_BASE}{ANALYSIS}")
    FOLDER = f"{FOLDER_BASE}{ANALYSIS}"
else:
    FOLDER = f"{FOLDER_BASE}{ANALYSIS}"

df = pd.read_csv('/home/dsilva/testes_ml/dataset/carol_correct/ep-eb_calibration.csv', sep=';', decimal=',')
df_val = pd.read_csv('/home/dsilva/testes_ml/dataset/carol_correct/ep_for_prediction.csv', sep=';', decimal=',')

# df = make_average(df, 2, 2)
# df_val = make_average(df_val, 2, 2)

# TRANSFORMAÇÕES

df_sg2_25 = sg(df, differentiation=2, window_size=25, spectra_start=2)
df_val_sg2_25 = sg(df_val, differentiation=2, window_size=25, spectra_start=2)

df_snv = snv(df, spectra_start=2)
df_val_snv = snv(df_val, spectra_start=2)

df_msc = msc(df, spectra_start=2)
df_val_msc = msc(df_val, spectra_start=2)

df_areanorm = area_norm(df, spectra_start=2)
df_val_areanorm = area_norm(df_val, spectra_start=2)

df_snv_sg11 = plus_sg(df, differentiation=1, window_size=11, spectra_start=2, transformation=snv, sg_first=False)
df_val_snv_sg11 = plus_sg(df_val, differentiation=1, window_size=11, spectra_start=2, transformation=snv, sg_first=False)

df_snv_sg2_11 = plus_sg(df, differentiation=2, window_size=11, spectra_start=2, transformation=snv, sg_first=False)
df_val_snv_sg2_11 = plus_sg(df_val, differentiation=2, window_size=11, spectra_start=2, transformation=snv, sg_first=False)

df_snv_sg2_25 = plus_sg(df, differentiation=2, window_size=25, spectra_start=2, transformation=snv, sg_first=False)
df_val_snv_sg2_25 = plus_sg(df_val, differentiation=2, window_size=25, spectra_start=2, transformation=snv, sg_first=False)

df_sg11_snv = plus_sg(df, differentiation=1, window_size=11, spectra_start=2, transformation=snv, sg_first=True)
df_val_sg11_snv = plus_sg(df_val, differentiation=1, window_size=11, spectra_start=2, transformation=snv, sg_first=True)

df_sg2_11_snv = plus_sg(df, differentiation=2, window_size=11, spectra_start=2, transformation=snv, sg_first=True)
df_val_sg2_11_snv = plus_sg(df_val, differentiation=2, window_size=11, spectra_start=2, transformation=snv, sg_first=True)

df_sg2_25_snv = plus_sg(df, differentiation=2, window_size=25, spectra_start=2, transformation=snv, sg_first=True)
df_val_sg2_25_snv = plus_sg(df_val, differentiation=2, window_size=25, spectra_start=2, transformation=snv, sg_first=True)

df_normalize_sg2_11 = plus_sg(df, differentiation=2, window_size=11, spectra_start=2, transformation=area_norm, sg_first=False)
df_val_normalize_sg2_11 = plus_sg(df_val, differentiation=2, window_size=11, spectra_start=2, transformation=area_norm, sg_first=False)

df_normalize_sg2_25 = plus_sg(df, differentiation=2, window_size=25, spectra_start=2, transformation=area_norm, sg_first=False)
df_val_normalize_sg2_25 = plus_sg(df_val, differentiation=2, window_size=25, spectra_start=2, transformation=area_norm, sg_first=False)

df_msc_sg2_11 = plus_sg(df, differentiation=2, window_size=11, spectra_start=2, transformation=msc, sg_first=False)
df_val_msc_sg2_11 = plus_sg(df_val, differentiation=2, window_size=11, spectra_start=2, transformation=msc, sg_first=False)

df_sg2_11_msc = plus_sg(df, differentiation=2, window_size=11, spectra_start=2, transformation=msc, sg_first=True)
df_val_sg2_11_msc = plus_sg(df_val, differentiation=2, window_size=11, spectra_start=2, transformation=msc, sg_first=True)

df_msc_sg2_25 = plus_sg(df, differentiation=2, window_size=25, spectra_start=2, transformation=msc, sg_first=False)
df_val_msc_sg2_25 = plus_sg(df_val, differentiation=2, window_size=25, spectra_start=2, transformation=msc, sg_first=False)

df_sg2_25_msc = plus_sg(df, differentiation=2, window_size=25, spectra_start=2, transformation=msc, sg_first=True)
df_val_sg2_25_msc = plus_sg(df_val, differentiation=2, window_size=25, spectra_start=2, transformation=msc, sg_first=True)



transformations = [(df, df_val, 'RAW'), 
                   (df_snv, df_val_snv, 'SNV'), 
                   (df_msc, df_val_msc, 'MSC'), 
                   (df_areanorm, df_val_areanorm, 'AREA_NORM'),
                   (df_snv_sg11, df_val_snv_sg11, 'SNV_SG11'), 
                   (df_snv_sg2_11, df_val_snv_sg2_11, 'SNV_SG2_11'), 
                   (df_sg2_25, df_val_sg2_25, 'SG2_25'), 
                   (df_snv_sg2_25, df_val_snv_sg2_25, 'SNV_SG2_25'), 
                   (df_sg11_snv, df_val_sg11_snv, 'SG11_SNV'), 
                   (df_sg2_11_snv, df_val_sg2_11_snv, 'SG2_11_SNV'), 
                   (df_sg2_25_snv, df_val_sg2_25_snv, 'SG2_25_SNV'), 
                   (df_msc_sg2_11, df_val_msc_sg2_11, 'MSC_SG2_11'), 
                   (df_sg2_11_msc, df_val_sg2_11_msc, 'SG2_11_MSC'), 
                   (df_msc_sg2_25, df_val_msc_sg2_25, 'MSC_SG2_25'), 
                   (df_sg2_25_msc, df_val_sg2_25_msc, 'SG2_25_MSC'), 
                   (df_normalize_sg2_11, df_val_normalize_sg2_11, 'AREA_NORM_SG2_11'), 
                   (df_normalize_sg2_25, df_val_normalize_sg2_25, 'AREA_NORM_SG2_25')]



results = np.zeros((len(transformations), 7))
results[:] = np.nan

df_results = pd.DataFrame(results)

for pos, transformation in enumerate(transformations[0:1]):
    rf = RandomForest(transformation[1], 100, 10, dataset_validation=transformation[1])
    #rf.search_hyperparameters(n_processors=-1, verbose=1, estimators=[100, 510, 100], min_samples_split=[2, 10, 100, 400], min_samples_leaf=[1,2, 3])
    rf.create_model()

    if save_results == True:
        save_results_of_model(rf, path=FOLDER, name=transformation[2], plots=True, out_table=True, variables_importance='random_forest')


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