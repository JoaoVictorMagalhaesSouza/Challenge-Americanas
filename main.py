#%% Loading libs
"""
    Essa seção é dedicada apenas para a importação das bibliotecas necessárias para
    resolução do desafio.
"""
from matplotlib.pyplot import axis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fastparquet

from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics, preprocessing
from feature_importance import plot_importance
from data_balancer import DataBalancer
from data_preparation import DataPreparation
from exploratory_analisys import ExploratoryAnalisys
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

#%% Configs
verbose = True
#%% Reading the input data
"""
    Aqui é dedicada uma seção para a leitura e visualização bruta e inicial dos dados
    no formato de tabela.
"""
input_data = pd.read_parquet('dataset_cdjr.parquet.gzip')
#display(input_data.head(10))
#%% Exploratory Analisys
'''
    Foi criada uma classe responsável apenas pela parte de Análise Exploratória dos Dados, denomidada
    ExploratoryAnalisys cujo 'construtor' recebe os dados de entrada.
'''
exploratory_analisys = ExploratoryAnalisys(input_data,verbose=verbose)
print(exploratory_analisys.describe_data())

'''
    Aqui visualizamos o comportamento 'temporal' das variáveis, já na tentativa de buscar insights mais visuais
    como outliers, etc...
'''
exploratory_analisys.view_time_series()
'''
    Posteriormente, crio os histogramas de cada uma das variáveis para entender melhor, e de maneira mais clara, como estão distribuídas.
'''
exploratory_analisys.view_histograms()
'''
    Na sequência, decido observar o nível de correlação das minhas variáveis de entrada com a minha target.
'''
exploratory_analisys.view_corr_plot()
'''
    Por fim, como se trata de um problema de classificação, vejo se temos um problema balanceado ou desbalanceado
'''
exploratory_analisys.view_target_distribuition()

'''
    Criei um balanceador de dados manual mas optei por não utilizá-lo nesta resolução.
'''
to_balance = False
if to_balance:
    balancer = DataBalancer(input_data)
    input_data = balancer.balance()

#%% Data Preparation
preprocess = DataPreparation(input_data)
old_input = input_data.copy()
input_data = preprocess.pipeline_pre_process()
N = 9
input_data = input_data.iloc[N: , :]
#%%
'''
    Criei novas features mas após testes, vi que não surtiram muito efeito
'''
features = ['feature0', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5',
       'feature6', 'feature7', 'feature8', 'feature9', 'feature10',
       'feature11', 'feature12', 'feature13', 'feature14', 'feature15']
features = input_data.drop(columns={'target'}).columns
#X_train, x_test, y_train, y_test = train_test_split(input_data.drop(columns={'target'}),input_data['target'],train_size=0.8)
X_train, x_test, y_train, y_test = train_test_split(input_data[features],input_data['target'],train_size=0.70)


# %% Catboost
ctb_model = CatBoostClassifier(iterations=150,
                                depth=8,
                                learning_rate=0.35,
                                loss_function='Logloss',
                                auto_class_weights='SqrtBalanced',
                                )
ctb_model.fit(X_train,y_train,plot=verbose)
ctb_predictions = ctb_model.predict(x_test)

print(classification_report(y_test,ctb_predictions))
'''
    Confusion Matrix
'''
if verbose:
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, ctb_predictions, labels=ctb_model.classes_, ax=ax, colorbar=False
    )
    plt.show()
print(f"Model acc: {accuracy_score(ctb_predictions,y_test)}")
print("ROC AUC CatBoost: ",metrics.roc_auc_score(ctb_predictions,y_test.values))

#%%
scores = cross_val_score(ctb_model, input_data[features],input_data['target'], cv=10)
print("Cross Validation (10) off Catboost: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# %% Feature Importance
plot_importance(ctb_model,features)
#%%