from transformations import sg
import pandas as pd

df = pd.read_csv('/home/dsilva/teste_pcr/foss_para_arff_calibracao.csv', sep=';', decimal=',')

df_sg = sg(df.iloc[:, 2:], 1, 25, 4)