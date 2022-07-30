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
from data_preparation import DataPreparation
from exploratory_analisys import ExploratoryAnalisys
from model_optimize import OptimizeCatboost
from split_data import SplitData
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV

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

#%% Data Preparation
preprocess = DataPreparation(input_data)
old_input = input_data.copy()
input_data = preprocess.pipeline_pre_process()
#%% Split data

# df_train = input_data.iloc[:(int(len(input_data)*0.7))]
# df_val = input_data.iloc[(int(len(input_data)*0.7)):(int(len(input_data)*0.85))]
# df_test = input_data.iloc[(int(len(input_data)*0.85)):]

# X_train, y_train = df_train.drop(columns={'target'}), df_train['target']
# x_val, y_val = df_val.drop(columns={'target'}), df_val['target']
# x_test, y_test = df_test.drop(columns={'target'}), df_test['target']
split_data = SplitData(input_data)

X_train, y_train, x_val, y_val, x_test, y_test = split_data.split_train_val_test()
#X_train, y_train, x_test, y_test = split_data.split_train_test()

# %% Creating model
params = {'depth': 7,
 'iterations': 400,
 'learning_rate': 0.1762341288441044,
 'loss_function': 'Logloss',
 'l2_leaf_reg': 2.433812145711232}
# ctb_model = CatBoostClassifier(iterations=200,
#                                 depth=8,
#                                 learning_rate=0.35,
#                                 loss_function='Logloss',
#                                 auto_class_weights='SqrtBalanced',
#                                 )
ctb_model = CatBoostClassifier(**params)
ctb_model.fit(X_train,y_train,plot=verbose)
#%% Validation 
ctb_predictions_val = ctb_model.predict(x_val)

print(classification_report(y_val,ctb_predictions_val))
'''
    Confusion Matrix for validation
'''
if verbose:
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_val, ctb_predictions_val, labels=ctb_model.classes_, ax=ax, colorbar=False
    )
    plt.show()
print(f"Model acc for validation: {accuracy_score(ctb_predictions_val,y_val)}")
print("ROC AUC for validation: ",metrics.roc_auc_score(ctb_predictions_val,y_val.values))

#%% Test
ctb_predictions_test = ctb_model.predict(x_test)

print(classification_report(y_test,ctb_predictions_test))
'''
    Confusion Matrix for test
'''
if verbose:
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, ctb_predictions_test, labels=ctb_model.classes_, ax=ax, colorbar=False
    )
    plt.show()
print(f"Model acc for validation: {accuracy_score(ctb_predictions_test,y_test)}")
print("ROC AUC for validation: ",metrics.roc_auc_score(ctb_predictions_test,y_test.values))

# %% Feature Importance
features = list(X_train.columns)
plot_importance(ctb_model,features)
#%% Optuna tunning
optimization = OptimizeCatboost(X_train,x_test,y_train,y_test)
best_params = optimization.optimize()

# %%
