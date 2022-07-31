import pandas as pd

class FeatureSelection():
    def __init__(self, input_data: pd.DataFrame, method='spearman', threshold=0.10):
        self.input_data = input_data
        self.method = method
        self.threshold = threshold
    
    def filter_features(self):
        if self.method == 'spearman':
            spearman_corr = abs(self.input_data.corr(method='spearman')['target'])
            best_vars = list(spearman_corr[spearman_corr>self.threshold].index)
            return self.input_data[best_vars]