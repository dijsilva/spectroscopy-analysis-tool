from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

from algorithms.regression import PCR, RandomForest, PLSR
from algorithms.classification import PCA_LDA
from transformations import snv, sg, msc, plus_sg, area_norm
import pandas as pd
from utils import make_average
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('/home/dsilva/teste_pcr/carol/ep-eb_calibration.csv', sep=';', decimal=',')
df_val = pd.read_csv('/home/dsilva/teste_pcr/carol/eb_for_prediction.csv', sep=';', decimal=',')

df_all = pd.read_csv('/home/dsilva/teste_pcr/foss_all_samples.csv', sep=';', decimal=',')
df_all_average = make_average(df_all, 2, 2)

# df_sg2_25 = sg(df, differentiation=2, window_size=25, spectra_start=2)
# df_val_sg2_25 = sg(df_val, differentiation=2, window_size=25, spectra_start=2)

# df_snv = snv(df, spectra_start=2)
# df_val_snv = snv(df_val, spectra_start=2)

# df_msc = msc(df, spectra_start=2)
# df_val_msc = msc(df_val, spectra_start=2)

# df_areanorm = area_norm(df, spectra_start=2)
# df_val_areanorm = area_norm(df_val, spectra_start=2)

# df_snv_sg2_25 = plus_sg(df, differentiation=2, window_size=25, spectra_start=2, transformation=snv, sg_first=False)
# df_val_snv_sg2_25 = plus_sg(df_val, differentiation=2, window_size=25, spectra_start=2, transformation=snv, sg_first=False)

# df_msc_sg2_25 = plus_sg(df, differentiation=2, window_size=25, spectra_start=2, transformation=msc, sg_first=False)
# df_val_msc_sg2_25 = plus_sg(df_val, differentiation=2, window_size=25, spectra_start=2, transformation=msc, sg_first=False)

# df_normalize_sg2_25 = plus_sg(df, differentiation=2, window_size=25, spectra_start=2, transformation=area_norm, sg_first=False)
# df_val_normalize_sg2_25 = plus_sg(df_val, differentiation=2, window_size=25, spectra_start=2, transformation=area_norm, sg_first=False)


# transformations = [(df, df_val), (df_sg2_25, df_val_sg2_25), (df_snv, df_val_snv), (df_msc, df_val_msc),
#                    (df_areanorm, df_val_areanorm), (df_snv_sg2_25, df_val_snv_sg2_25), (df_msc_sg2_25, df_val_msc_sg2_25), 
#                    (df_normalize_sg2_25, df_val_normalize_sg2_25)]

"""
RMSE = []
R2 = []

for transformation in transformations:
    rf = RandomForest(transformation[0], 400, 10, dataset_validation=transformation[1], rf_max_depth=10, max_features_rf='sqrt')
    rf.create_model()

    RMSE.append(rf.metrics['validation']['RMSE'])
    R2.append(rf.metrics['validation']['R2'])
    #rf.save_results(path='/home/dsilva/teste_random_forest', plots=True)"""


#rf = RandomForest(df_all, 350, 'loo', dataset_validation=df_val, rf_max_depth=40)
rf = RandomForest(df, 100, 10, dataset_validation=df_val, rf_max_depth=40, rf_oob_score=True)
rf.calibrate()
rf.validate()
rf.save_results(path="/home/dsilva/teste_pcr/results", plots=True, name=nir)
#rf.search_hyperparameters(verbose=2, n_processors=-1, max_depth=[10, 50, 5], estimators=[50, 500, 50], 
#                          min_samples_leaf=[1, 2, 3], max_features=['sqrt', 'auto'], bootstrap=[True, False], min_samples_split=[2, 3])

#best_params = rf._best_params

# {'bootstrap': False, 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 150}
#{'bootstrap': True, 'max_depth': 40, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 350}