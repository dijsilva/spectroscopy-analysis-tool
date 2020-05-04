from algorithms import PLSR
from transformations import sg, snv, plus_sg, area_norm
import pandas as pd

df = pd.read_csv('/home/dsilva/teste_pcr/foss_para_arff_calibracao.csv', sep=';', decimal=',')
df_validacao = pd.read_csv('/home/dsilva/teste_pcr/foss_para_arff_validacao.csv', sep=';', decimal=',')

# df_sg = df.copy()
# df_sg = sg(df.iloc[:, 2:], 1, 25, 4)

df_area_norm = area_norm(df)


#plsr = PLSR(df, 20, 'loo', dataset_validation=df_validacao)

