from algorithms.regression import PCR, RandomForest, PLSR
from algorithms.classification import PCA_LDA
from transformations import snv, sg, msc
import pandas as pd
from utils import make_average
import matplotlib.pyplot as plt

# regression
df = pd.read_csv('/home/dsilva/teste_pcr/foss_para_arff_calibracao.csv', sep=';', decimal=',')
df_val = pd.read_csv('/home/dsilva/teste_pcr/foss_para_arff_validacao.csv', sep=';', decimal=',')

#df_average = make_average(df, 2, 2)
#df_val_average = make_average(df_val, 2, 2)


rf = RandomForest(df, 100, 'loo', dataset_validation=df_val)
rf.create_model()


#r2 = []
#rmse = []
# for i in range(2, 21):
#pcr = PCR(df, 20, 'loo', dataset_validation=df_val)
#pcr.create_model()
# r2.append(pcr.metrics['validation']['R2'])
# rmse.append(pcr.metrics['validation']['RMSE'])
# print(f"{i} componentes")



#classification
# df = pd.read_csv('/home/dsilva/teste_pcr/espectros_frutos_polpa.csv', sep=';', decimal=',')
# df_test = pd.read_csv('/home/dsilva/teste_pcr/espectros_frutos_polpa1.csv', sep=';', decimal=',')

# df_average = make_average(df, 3, 2)
# df_test_average = make_average(df_test, 3, 2)

# lda = PCA_LDA(df_average, 8, 'loo', data_validation=df_test_average)
# lda.create_model()