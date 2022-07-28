import pandas as pd


class DataBalancer():
    def __init__(self, input_data):
        self.input_data = input_data
    
    def balance(self):
        number_of_occurrencies_0 = self.input_data['target'].value_counts()[0]
        number_of_occurrencies_1 = self.input_data['target'].value_counts()[1]
        diference = number_of_occurrencies_1 - number_of_occurrencies_0
        indexes_of_occurrencies_1 = list(self.input_data[self.input_data['target']==1].index)
        indexes_of_occurrencies_0 = list(self.input_data[self.input_data['target']==0].index)
        indexes_of_occurrencies_1 = indexes_of_occurrencies_1[diference:]
        
        new_indexes = indexes_of_occurrencies_0 + indexes_of_occurrencies_1

        output = self.input_data[self.input_data.index.isin(new_indexes)]
        return output
