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
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import feature_engineering as fe
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from limpeza import CleansingData
from catboost import CatBoostClassifier
#%% Reading the input data
"""
    Aqui é dedicada uma seção para a leitura e visualização bruta e inicial dos dados
    no formato de tabela.
"""
input_data = pd.read_parquet('dataset_cdjr.parquet.gzip')
#display(input_data.head(10))
#%% Exploratory Analisys
'''
    Essa primeira analise será um gráfico de linhas para visualizar, de modo geral, o comportamento das variáveis ao longo do dataframe de entrada.
    Para isso, criei uma nova coluna chamada x-axis para simular o passar do tempo.
'''
#display(input_data.describe())
analisys = False
if analisys:
    input_data['x-axis'] = [x for x in range(len(input_data))]
    for column in input_data.columns:
        if column not in['x-axis','target']:
            fig = px.line(data_frame=input_data,x='x-axis',y=column)
            fig.show()
    input_data.pop('x-axis')
#input_data = input_data.sample(frac=1)
'''
    Prossigo então fazendo uma análise de correlação linear e de informação mútua das variáveis de entrada com a nossa target
'''
pearson_df = abs(input_data.corr('pearson'))['target']
'''
    De cara observo que as correlações com a variável alvo são bem baixas. Neste caso, eu não opto por tirar nenhuma variável e, sendo assim,
    mantenho todas as variáveis de entrada.
'''

#%% Data Preparation
'''
    Como as variáveis possuem uma fraca correlação com a target, ou seja, me dão pouca informação acerca da minha variável alvo,
    tento criar novas variáveis para enriquecer o nível de informação dos dados (Feature Engineering).
# '''
# new_features = fe.FeatureEngineering(input_data).pipeline_feat_eng()
# input_data = input_data.merge(new_features,on=input_data.index,how='left')
#input_data = input_data.fillna(0)
#%% Split data
clean = False
if clean:
    cleansing = CleansingData(input_data)
    input_data = cleansing.remove_outliers()
    #input_data = input_data.interpolate(method='polynomial',order=2)
    #input_data = input_data.interpolate(method='linear')
    #input_data = input_data.fillna(method='bfill').fillna(method='ffill')

X_train, x_test, y_train, y_test = train_test_split(input_data.drop(columns={'target'}),input_data['target'],train_size=0.8)
#%% Create model
xgb_model = XGBClassifier(max_depth = 7,
                         n_estimators=1500,
                         learning_rate=0.2,
                         subsample=0.6,
                         colsample_bytree=0.7,
                         )
xgb_model.fit(X_train,y_train)
xgb_predictions = xgb_model.predict(x_test)
#%% Evaluate model
print(f"Model acc: {accuracy_score(xgb_predictions,y_test)}")
print(classification_report(y_test,xgb_predictions))
'''
    Confusion Matrix
'''
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test, xgb_predictions, labels=xgb_model.classes_, ax=ax, colorbar=False
)
plt.show()
# %% Cross validation
scores = cross_val_score(xgb_model, input_data.drop(columns={'target'}),input_data['target'], cv=10)
print("Accuracy XGBoost: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# %% Catboost
ctb_model = CatBoostClassifier(iterations=150,depth=7,learning_rate=0.3)
ctb_model.fit(X_train,y_train)
ctb_predictions = ctb_model.predict(x_test)
print(f"Model acc: {accuracy_score(ctb_predictions,y_test)}")
print(classification_report(y_test,ctb_predictions))
'''
    Confusion Matrix
'''
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test, ctb_predictions, labels=ctb_model.classes_, ax=ax, colorbar=False
)
plt.show()
scores = cross_val_score(ctb_model, input_data.drop(columns={'target'}),input_data['target'], cv=10)
print("Accuracy Catboost: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# %%
