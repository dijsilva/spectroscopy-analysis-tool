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
#df_all = pd.read_csv('/home/dsilva/teste_pcr/foss_all_samples.csv', sep=';', decimal=',')
#df_msc_sg2_25 = plus_sg(df, True, transformation=msc, differentiation=2, window_size=25, spectra_start=2)
#df_val_msc_sg2_25 = plus_sg(df_val, True, transformation=msc, differentiation=2, window_size=25, spectra_start=2)


rf = RandomForest(df, 400, 10, dataset_validation=df_val, rf_max_depth=10, max_features_rf='sqrt')
rf.create_model()
rf.save_results(path='/home/dsilva/teste_random_forest', plots=True)
#rf.search_hyperparameters(verbose=2, n_processors=2, max_depth=[10, 50, 5], min_samples_leaf=[1, 2, 3], max_features=['sqrt', 'auto'])