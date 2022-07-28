import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

class FeatureEngineering():
    def __init__(self, df: pd.DataFrame):
        self.data = df.copy()
        self.df_fe = pd.DataFrame()
        self.target = 'target'

    def derivada(self):        
        for column in self.data.columns:
            if column != self.target:
                 self.df_fe[f'{column}_derivative'] = self.data[column].diff()

    def integral(self):
        for column in self.data.columns:
            if column != self.target:
                
                self.df_fe[f'{column}_integral'] = self.data[column].rolling(10).sum()

    def momentos_estatisticos(self):
        for column in self.data.columns:
            if column != self.target:
                
                self.df_fe[f'{column}_moving_average'] = self.data[column].rolling(10).mean()
                self.df_fe[f'{column}_std'] = self.data[column].rolling(10).std()


    def pipeline_feat_eng(self):
        self.derivada()
        self.integral()
        self.momentos_estatisticos()
    
        return self.df_fe.copy()