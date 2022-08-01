#%% Loading libs
"""
    Essa seção é dedicada apenas para a importação das bibliotecas necessárias para
    resolução do desafio.
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
from utils.feature_importance import plot_importance
from utils.data_preparation import DataPreparation
from utils.exploratory_analisys import ExploratoryAnalisys
from utils.model_optimize import OptimizeModel
from utils.split_data import SplitData
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import joblib

#%% Configs
'''
    Verbose = True: Plotagem dos gráficos.
    Verbose = False: Não plotagem dos gráficos.
'''
verbose = False
#%% Reading the input data
"""
    Aqui é dedicada uma seção para a leitura e visualização bruta e inicial dos dados
    no formato de tabela.
"""
input_data = pd.read_parquet('data/dataset_cdjr.parquet.gzip')

#%% Exploratory Analisys
'''
    Foi criada uma classe responsável apenas pela parte de Análise Exploratória dos Dados, denomidada
    ExploratoryAnalisys cujo 'construtor' recebe os dados de entrada.
'''
exploratory_analisys = ExploratoryAnalisys(input_data,verbose=verbose)

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
X_train, y_train, x_test, y_test = split_data.split_train_test()

# %% Creating model

forest = RandomForestClassifier(n_estimators=100,
                                criterion='entropy',
                                max_depth=4,
                                min_samples_leaf=10,
                                random_state=42,
)
#%% Evaluation model with KFolding

f1_scores = []
auc_scores = []
for fold in splited_data.keys():
    print(fold)
    #ctb_model.fit(splited_data[fold]['X_train'],splited_data[fold]['y_train'],plot=verbose, verbose=verbose)
    forest.fit(splited_data[fold]['X_train'],splited_data[fold]['y_train'])
    print(f"    => Score for train: {forest.score(splited_data[fold]['X_train'],splited_data[fold]['y_train'])}")
    forest_predictions_test = forest.predict(splited_data[fold]['X_test'])
    if verbose:
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(
            splited_data[fold]['y_test'], forest_predictions_test, labels=forest.classes_, ax=ax, colorbar=False
        )
        plt.show()
        plt.savefig(f'figures/conf_matrix_{fold}.png')
    #print(f"    => Model acc for test: {accuracy_score(forest_predictions_test,splited_data[fold]['y_test'])}")
    print('     => F1 score for test: ',f1_score(forest_predictions_test,splited_data[fold]['y_test']))
    #print("     => ROC AUC for test: ",metrics.roc_auc_score(forest_predictions_test,splited_data[fold]['y_test'].values))
    auc_scores.append(metrics.roc_auc_score(forest_predictions_test,splited_data[fold]['y_test'].values))
    f1_scores.append(f1_score(forest_predictions_test,splited_data[fold]['y_test']))
    print('')
auc_scores = pd.Series(auc_scores)
f1_scores = pd.Series(f1_scores)
#print("ROC AUC score for CV: %0.4f (+/- %0.2f)" % (auc_scores.mean(), auc_scores.std() * 2))
print("Average F1 score: %0.4f (+/- %0.2f)" % (f1_scores.mean(), f1_scores.std()))
print('')
print('')
# %% Ploting the Feature Importance
verbose=True
if verbose:
    forest = forest.fit(X_train,y_train)
    features = list(X_train.columns)
    plot_importance(forest,features)
#%% Optuna tunning

to_optimize = False
if to_optimize:
    optimization = OptimizeModel(X_train,x_test,y_train,y_test, splited_data)
    best_params = optimization.optimize()

#%% Evaluation for fix split
print("Evaluation for sequencial data (first 80% for train and last 20% for test)")
model2 = forest
model2.fit(X_train,y_train)
print(f"    => Score for train: {model2.score(X_train,y_train)}")
predicts = model2.predict(x_test)
print(f"    => F1 score for test: {f1_score(predicts,y_test)}")
#print(f'    => ROC AUC score for test: {metrics.roc_auc_score(predicts,y_test)}')
print('')
print('')
# %% Exporting model
to_save = False
if to_save:
    print("Saving model...")
    print('')
    print('')
    path = 'saved_model/joao_victor_random_forest.sav'
    joblib.dump(model2,path)
#%% Testing predictions for saved model
to_load = False
if to_load:
    print("Loading model...")
    print('')
    print('')
    print("Evaluation for saved model with sequencial data: ")
    loaded_model = joblib.load(path)
    predictions = loaded_model.predict(x_test)
    print(f"    => Score for train: {loaded_model.score(X_train,y_train)}")
    print(f"    => F1 score for saved model: {f1_score(predictions,y_test)}")
    #print(f'    => ROC AUC score for saved model: {metrics.roc_auc_score(predictions,y_test)}')
# %%
