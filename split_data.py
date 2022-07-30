import pandas as pd
from sklearn.model_selection import StratifiedKFold

class SplitData():
    def __init__(self, input_data:pd.DataFrame):
        self.input_data = input_data
    
    def split_train_val_test(self, train_size=0.7, val_size=0.15):
        df_train = self.input_data.iloc[:(int(len(self.input_data)*train_size))]
        df_val = self.input_data.iloc[(int(len(self.input_data)*train_size)):(int(len(self.input_data)*(train_size+val_size)))]
        df_test = self.input_data.iloc[(int(len(self.input_data)*(train_size+val_size))):]

        X_train, y_train = df_train.drop(columns={'target'}), df_train['target']
        x_val, y_val = df_val.drop(columns={'target'}), df_val['target']
        x_test, y_test = df_test.drop(columns={'target'}), df_test['target']

        return X_train, y_train, x_val, y_val, x_test, y_test

    def split_train_test(self, train_size=0.8):
        df_train = self.input_data.iloc[:(int(len(self.input_data)*train_size))]
        df_test = self.input_data.iloc[(int(len(self.input_data)*train_size)):]

        X_train, y_train = df_train.drop(columns={'target'}), df_train['target']
        x_test, y_test = df_test.drop(columns={'target'}), df_test['target']

        return X_train, y_train, x_test, y_test
    
    def split_kfold(self, num_folds=4):
        dict_data = {}
        data_aux = self.input_data.copy()
        data_aux.index = [x for x in range(len(data_aux))]
        X = data_aux.drop(columns={'target'})
        y = data_aux['target']
        i = 0
        skf = StratifiedKFold(n_splits=num_folds,shuffle=False)
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[X.index.isin(train_index)], X[X.index.isin(test_index)]
            y_train, y_test = y[y.index.isin(train_index)], y[y.index.isin(test_index)]
            dict_data[f'FOLD {i}'] = {}
            dict_data[f'FOLD {i}']['X_train'] = X_train
            dict_data[f'FOLD {i}']['X_test'] = X_test
            dict_data[f'FOLD {i}']['y_train'] = y_train
            dict_data[f'FOLD {i}']['y_test'] = y_test
            i+=1
        
        return dict_data