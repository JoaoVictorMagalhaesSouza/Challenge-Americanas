import pandas as pd
from copy import deepcopy
import numpy as np
class CleansingData():
    def __init__(self, input_data: pd.DataFrame):
        self.input_data = input_data
    
    def remove_outliers(self):
        output_data = deepcopy(self.input_data)
        for column in output_data.columns:
            if column != 'target':
                Q1 = output_data[column].quantile(0.25)
                Q3 = output_data[column].quantile(0.75)
                IQR = Q3 - Q1
                IQR_ratio = 2
                output_data[column] = np.where(((output_data[column] < (Q1 - IQR_ratio*IQR))|(output_data[column] > (Q3 + IQR_ratio*IQR))),output_data[column].mean(),output_data[column])
    
        return output_data

