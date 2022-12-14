import pandas as pd
from copy import deepcopy
import plotly.express as px
import plotly.graph_objects as go

class ExploratoryAnalisys():
    def __init__(self,input_data,verbose=True):
        self.input_data = input_data
        self.verbose = verbose
        
    def describe_data(self):
        return self.input_data.describe()

    def view_time_series(self):
        visualization_data = deepcopy(self.input_data)
        visualization_data['x-axis'] = [x for x in range(len(visualization_data))]
        for column in visualization_data.columns:
            if column not in['x-axis','target']:
                fig = px.line(data_frame=visualization_data,x='x-axis',y=column,
                labels={
                    'x-axis': "Amostra",
                    column: "Valor"
                },
                title=f"Comportamento da variável {column}"
                )
                if self.verbose:
                    fig.show()
                    fig.write_image(f'figures/time_series_{column}.png')
                    
        visualization_data.pop('x-axis')
    
    def view_histograms(self):
        visualization_data = deepcopy(self.input_data)
        for column in visualization_data.columns:
            if column not in['target']:
                fig = go.Figure(data=[go.Histogram(x=visualization_data[column])])
                fig.update_layout(
                    title_text = f'Distribuição de valores da variável {column}',
                    xaxis_title_text='Valor', 
                    yaxis_title_text='Quantidade', 
                    bargap=0.2, 
                    bargroupgap=0.1 
                )
                if self.verbose:
                    fig.show()
                    fig.write_image(f'figures/histogram_{column}.png')


    def view_corr_plot(self):
        pearson_df = abs(self.input_data.corr('pearson'))['target']
        #spearman_df = abs(self.input_data.corr('spearman'))['target']
        df_corr = pd.DataFrame()
        df_corr['Pearson'] = pearson_df.values
        #df_corr['Spearman'] = spearman_df.values
        df_corr['Variável'] = pearson_df.index

        df_corr = df_corr[df_corr['Variável']!='target']
            
        fig = go.Figure(
            data = [go.Bar(
                name = 'Corr. Pearson',
                x = df_corr['Variável'],
                y = df_corr['Pearson']

            ),
            # go.Bar(
            #     name = 'Corr. Spearman',
            #     x = df_corr['Variável'],
            #     y = df_corr['Spearman'] #?
            # )
            
            ]
        )
        fig.update_layout(
                    title_text = 'Gráfico de correlação das variáveis de entrada com a target',
                    xaxis_title_text='Variável', 
                    yaxis_title_text='Nível de correlação absoluta', 
                    
                )
        if self.verbose:
            fig.show()
            fig.write_image(f'figures/correlation_plot.png')

    def view_target_distribuition(self):
        counts_0 = self.input_data['target'].value_counts()[0]
        counts_1 = self.input_data['target'].value_counts()[1]
        fig = go.Figure(
            data = [go.Bar(
                name = 'Classe 0',
                x = ['0'],
                y = [counts_0]

            ),
            go.Bar(
                name = 'Classe 1',
                x = ['1'],
                y = [counts_1]
            )
            
            ]
        )
        fig.update_layout(
                    title_text = 'Gráfico de distribuição da target',
                    xaxis_title_text='Classe', 
                    yaxis_title_text='Número de ocorrências', 
                    
                )
        if self.verbose:        
            fig.show()
            fig.write_image(f'figures/target_distribuition.png')
        

    def view_boxplot(self):
        for column in self.input_data.columns:
            if column not in['target']:
                fig = px.box(self.input_data, y=column)
                fig.update_layout(
                title_text = f'Boxplot da variável {column}',
                yaxis_title_text='Valor', 
                
            )
            if self.verbose:        
                fig.show()
                fig.write_image(f'figures/box_plot_{column}.png')

