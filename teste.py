from algorithms.regression import PCR
from algorithms.classification import PCA_LDA
from transformations import snv
import pandas as pd
import matplotlib.pyplot as plt

# regression
#df = pd.read_csv('/home/dsilva/teste_pcr/foss_para_arff_calibracao.csv', sep=';', decimal=',')
#df_val = pd.read_csv('/home/dsilva/teste_pcr/foss_para_arff_validacao.csv', sep=';', decimal=',')

#r2 = []
#rmse = []
# for i in range(2, 21):
#pcr = PCR(df, 20, 'loo', dataset_validation=df_val)
#pcr.create_model()
# r2.append(pcr.metrics['validation']['R2'])
# rmse.append(pcr.metrics['validation']['RMSE'])
# print(f"{i} componentes")



#classification
df = pd.read_csv('/home/dsilva/teste_pcr/espectros_todos_os_frutos_polpa.csv', sep=';', decimal=',')
df1 = pd.read_csv('/home/dsilva/teste_pcr/espectros_frutos_polpa.csv', sep=';', decimal=',')

#df_snv = snv(df)

lda = PCA_LDA(df1, 9)
lda.create_model()
a = lda.plot_confusion_matrix()


