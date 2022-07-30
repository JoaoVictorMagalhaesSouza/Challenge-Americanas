#%% Loading libs
"""
    Essa seção é dedicada apenas para a importação das bibliotecas necessárias para
    resolução do desafio.
"""
from posixpath import split, splitdrive
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
from sklearn.ensemble import RandomForestClassifier


#%% Configs
verbose = False
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
    Findando visualizar melhor algumas estatísticas sobre as variáveis, construo os boxplots delas.
'''
exploratory_analisys.view_boxplot()

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
split_data = SplitData(input_data)
splited_data = split_data.split_kfold(num_folds=10)
#X_train, y_train, x_val, y_val, x_test, y_test = split_data.split_train_val_test()
X_train, y_train, x_test, y_test = split_data.split_train_test()

# %% Creating model
params = {'depth': 4,
 'iterations': 80,
 'learning_rate': 0.0641091951763684,
 'loss_function': 'CrossEntropy',
 'min_data_in_leaf': 8}

ctb_model = CatBoostClassifier(**params)

forest = RandomForestClassifier(n_estimators=100,
                                criterion='entropy',
                                max_depth=7,
                                min_samples_leaf=10,
                                random_state=42,
                                                              
                                )

#%% Evaluation - RANDOM FOREST
scores = []
for fold in splited_data.keys():
    print(fold)
    #ctb_model.fit(splited_data[fold]['X_train'],splited_data[fold]['y_train'],plot=verbose, verbose=verbose)
    forest.fit(splited_data[fold]['X_train'],splited_data[fold]['y_train'])
    print(f"    => Score de treino: {forest.score(splited_data[fold]['X_train'],splited_data[fold]['y_train'])}")
    forest_predictions_test = forest.predict(splited_data[fold]['X_test'])
    if verbose:
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(
            splited_data[fold]['y_test'], forest_predictions_test, labels=forest.classes_, ax=ax, colorbar=False
        )
        plt.show()
    print(f"    => Model acc for test: {accuracy_score(forest_predictions_test,splited_data[fold]['y_test'])}")
    print("     => ROC AUC for test: ",metrics.roc_auc_score(forest_predictions_test,splited_data[fold]['y_test'].values))
    scores.append(metrics.roc_auc_score(forest_predictions_test,splited_data[fold]['y_test'].values))

scores = pd.Series(scores)
print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#%% Evaluation - Catboost
scores = []
for fold in splited_data.keys():
    print(fold)
    ctb_model.fit(splited_data[fold]['X_train'],splited_data[fold]['y_train'],plot=verbose, verbose=verbose)
    print(f"    => Score de treino: {ctb_model.score(splited_data[fold]['X_train'],splited_data[fold]['y_train'])}")
    ctb_predictions_test = ctb_model.predict(splited_data[fold]['X_test'])
    verbose = False
    if verbose:
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(
            splited_data[fold]['y_test'], ctb_predictions_test, labels=ctb_model.classes_, ax=ax, colorbar=False
        )
        plt.show()
    print(f"    => Model acc for test: {accuracy_score(ctb_predictions_test,splited_data[fold]['y_test'])}")
    print("     => ROC AUC for test: ",metrics.roc_auc_score(ctb_predictions_test,splited_data[fold]['y_test'].values))
    scores.append(metrics.roc_auc_score(ctb_predictions_test,splited_data[fold]['y_test'].values))

scores = pd.Series(scores)
print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# %% Feature Importance
features = list(X_train.columns)
plot_importance(ctb_model,features)
#%% Optuna tunning
optimization = OptimizeCatboost(X_train,x_test,y_train,y_test)
best_params = optimization.optimize()

# %% Evaluation for fix split
model2 = forest
model2.fit(X_train,y_train)
print(f"Score de treino: {model2.score(X_train,y_train)}")
predicts = model2.predict(x_test)
print(f"Score de teste: {model2.score(x_test,y_test)}")
# %%
