import pandas as pd
from cleaning import CleansingData
from feature_engineering import FeatureEngineering
import random
from feature_selection import FeatureSelection

class DataPreparation():
    def __init__(self,input_data: pd.DataFrame):
        self.input_data = input_data
        self.output_data = pd.DataFrame()
        self.dataCleansing = CleansingData(self.input_data)
        self.featureEngineering = FeatureEngineering(self.input_data)
        self.featureSelection = FeatureSelection(self.input_data)
    
    
    def clean_data(self):
        '''
            Observando a distribuição das variáveis, apenas uma delas (feature 3) se assemelha
            a uma distribuição Gaussiana (normal). Logo, o método de limpeza mais indicado seria por
            IQR.
        '''
        self.output_data = self.dataCleansing.remove_outliers()
        self.output_data = self.output_data.dropna() 
    
    def create_new_features(self):
        new_features = self.featureEngineering.pipeline_feat_eng()
        self.output_data = self.output_data.merge(new_features,on=self.output_data.index,how='inner',sort=False)
        self.output_data.index = self.output_data['key_0'].values
        self.output_data.pop('key_0')
        self.output_data = self.output_data.dropna()
    
    def normalize_data(self, method='std'):
        target = self.output_data.pop('target')
        if method=='min-max':
            self.output_data = (self.output_data-self.output_data.min())/(self.output_data.max()-self.output_data.min())
        elif method=='std':
            self.output_data = (self.output_data-self.output_data.mean())/(self.output_data.std())

        self.output_data['target'] = target
    
    def shuffle_data(self):
        self.output_data = self.output_data.sample(frac=1)

    def feat_select(self):
        self.output_data = self.featureSelection.filter_features()
    
    def pipeline_pre_process(self):
        self.clean_data()
        self.create_new_features()
        #self.feat_select()
        
        return self.output_data
