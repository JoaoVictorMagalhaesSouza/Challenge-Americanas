import pandas as pd
from utils.cleaning import CleansingData
from utils.feature_engineering import FeatureEngineering
from utils.feature_selection import FeatureSelection
from copy import deepcopy

class DataPreparation():
    def __init__(self,input_data: pd.DataFrame):
        self.input_data = input_data
        self.output_data = deepcopy(input_data)
        self.dataCleansing = CleansingData(self.input_data)
        self.featureEngineering = FeatureEngineering(self.input_data)
        self.featureSelection = FeatureSelection(self.input_data)
    
    
    def clean_data(self):
        '''
            Observando a distribuição das variáveis, apenas uma delas (feature 3) se assemelha
            a uma distribuição Gaussiana (normal). Logo, o método de limpeza mais indicado seria por
            IQR.
        '''
        print("     => Realizando a limpeza...")
        self.output_data = self.dataCleansing.remove_outliers()
        self.output_data = self.output_data.dropna() 
    
    def create_new_features(self):
        print("     => Criando novas features...")
        new_features = self.featureEngineering.pipeline_feat_eng()
        self.output_data = self.output_data.merge(new_features,on=self.output_data.index,how='inner',sort=False)
        self.output_data.index = self.output_data['key_0'].values
        self.output_data.pop('key_0')
        self.output_data = self.output_data.dropna()
    
    def normalize_data(self, method='std'):
        print("     => Normalizando os dados...")
        target = self.output_data.pop('target')
        if method=='min-max':
            self.output_data = (self.output_data-self.output_data.min())/(self.output_data.max()-self.output_data.min())
        elif method=='std':
            self.output_data = (self.output_data-self.output_data.mean())/(self.output_data.std())

        self.output_data['target'] = target
    
    def shuffle_data(self):
        print("     => Aplicando shuffle nos dados...")
        self.output_data = self.output_data.sample(frac=1,random_state=42)

    def feat_select(self):
        print("     => Aplicando a etapa de feature selection...")
        self.output_data = self.featureSelection.filter_features()
    
    def pipeline_pre_process(self):
        self.clean_data()
        self.create_new_features()
        #self.shuffle_data()
        #self.feat_select()
        #self.normalize_data()
        
        return self.output_data
