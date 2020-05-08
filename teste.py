from algorithms.regression import PCR, RandomForest, PLSR
from algorithms.classification import PCA_LDA
from transformations import snv, sg, msc, plus_sg, area_norm
import pandas as pd
from utils import make_average
import matplotlib.pyplot as plt

# regression
df = pd.read_csv('/home/dsilva/teste_pcr/foss_para_arff_calibracao.csv', sep=';', decimal=',')
df_val = pd.read_csv('/home/dsilva/teste_pcr/foss_para_arff_validacao.csv', sep=';', decimal=',')

#df_average = make_average(df, 2, 2)
#df_val_average = make_average(df_val, 2, 2)

df_transformed = plus_sg(df, spectra_start=2,transformation=msc, differentiation=2, window_size=25, sg_first=False)
df_val_transformed = plus_sg(df_val, spectra_start=2, transformation=msc, differentiation=2, window_size=25, sg_first=False)
#df_transformed = sg(df, spectra_start=2, differentiation=2, window_size=25)
#df_val_transformed = sg(df_val, spectra_start=2,  differentiation=2, window_size=25)


rf = RandomForest(df_transformed, 100, 'loo', dataset_validation=df_val_transformed)
rf.create_model()


print('Validation')
print(rf.metrics['validation']['R2'])
print(rf.metrics['validation']['RMSE'])

print('Cross-Validation')
print(rf.metrics['cross_validation']['R2'])
print(rf.metrics['cross_validation']['RMSE'])