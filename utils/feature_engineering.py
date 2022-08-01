import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

class FeatureEngineering():
    def __init__(self, df: pd.DataFrame):
        self.data = df.copy()
        self.df_fe = pd.DataFrame()
        self.target = 'target'

    def derivada(self):
        print("         => Criando a derivada das variáveis...")        
        for column in self.data.columns:
            if column != self.target:
                 self.df_fe[f'{column}_derivative'] = self.data[column].diff()

    def integral(self):
        print("         => Criando a integral das variáveis...")
        for column in self.data.columns:
            if column != self.target:
                
                self.df_fe[f'{column}_integral'] = self.data[column].rolling(5).sum()

    def momentos_estatisticos(self):
        print("         => Criando os momentos estatísticos móveis das variáveis...")
        for column in self.data.columns:
            if column != self.target:
                
                self.df_fe[f'{column}_moving_average'] = self.data[column].rolling(5).mean()
                self.df_fe[f'{column}_std'] = self.data[column].rolling(5).std()


    def combinacoes_polinomiais(self):
        print("         => Criando as combinações polinomiais das variáveis...")
        df_poly = PolynomialFeatures(2)
        cols = self.data.columns
        cols = cols[0:len(cols)-2]
        df_poly = pd.DataFrame(df_poly.fit_transform(self.data[cols]))
        qtde_colunas = len(df_poly.columns)
        df_poly = df_poly.drop(columns=[x for x in range (len(cols)+1)])
        nome_novas_colunas = []
        nao_vistadas = list(cols.copy())
        for coluna in cols:
            
            atual = coluna
            nome_novas_colunas.append(f'{coluna}^2')
            for _ in nao_vistadas:
                if (_ != atual):
                    nome_novas_colunas.append(f'{coluna}*{_}')
            
            nao_vistadas.remove(coluna)
        
        nome_velhas_colunas = [x for x in range(len(cols)+1,qtde_colunas)]
        for i in range(nome_velhas_colunas[0],nome_velhas_colunas[-1]+1):
            df_poly = df_poly.rename(columns={i:nome_novas_colunas[i-nome_velhas_colunas[0]]})

        for col in df_poly.columns:
            self.df_fe[col] = df_poly[col].values

    def pipeline_feat_eng(self):
        self.derivada()
        self.integral()
        self.momentos_estatisticos()
        self.combinacoes_polinomiais()
        return self.df_fe.copy()