import pandas as pd

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