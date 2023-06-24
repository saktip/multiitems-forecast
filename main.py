import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_plotly, plot_components_plotly
import numpy as np
import math

#untuk tuning (pencarian) parameter terbaik
import itertools

class Preparing():
    @staticmethod
    def get_unique_items(df, item_column_name):
        list_item = df[item_column_name].unique() 
        return list_item

class Tuning():   
    param_grid = {  
        'changepoint_prior_scale': [],
        'seasonality_prior_scale': []
    }
    
    list_items = []
    tuning_results = {}
    
    def __init__(self, param_grid, df, item_column_name):
        self.param_grid = param_grid
        self.df = df
        self.item_column_name = item_column_name
        self.list_items = Preparing().get_unique_items(df=self.df, item_column_name='kode')
        
    def proccess(self):
        all_params = [dict(zip(self.param_grid.keys(), v)) for v in itertools.product(*self.param_grid.values())]

        maes = {} # Store the MAE for each params here
        temp_mae = {}
        # tuning_results = {}

        for k in range(len(self.list_items)):
            df_temp = self.df[(self.df[self.item_column_name] == self.list_items[k])].copy()
            df_temp['cap'] = 50
            df_temp['floor'] = 0
            # df_temp = df_temp[['ds','y','cap', 'floor']]
            temp_mae[self.list_items[k]] = []
            print(self.list_items[k])
            for params in all_params:
                #fb prophet ini "will by default fit weekly and yearly seasonalities"
                m = Prophet(**params, growth='logistic').fit(df_temp) #, growth='logistic'
                df_cv = cross_validation(m, initial='7 days', horizon='30 days', parallel="processes")
                df_p = performance_metrics(df_cv, rolling_window=0) #kalo muncul assertion error untuk growth='logistic', rolling window isi 0 (coba highlight fungsinya)
                temp_mae[self.list_items[k]].append(df_p['mae'].values[0])
                maes[self.list_items[k]] = temp_mae[self.list_items[k]]
                
            self.tuning_results[self.list_items[k]] = pd.DataFrame(all_params)
            self.tuning_results[self.list_items[k]]['mae'] = pd.DataFrame(maes[self.list_items[k]])
            self.tuning_results[self.list_items[k]] = self.tuning_results[self.list_items[k]].sort_values(['mae'])
        return self.tuning_results

# class ForecastWithParam():


# https://stackoverflow.com/questions/64961448/python-pandas-warning-a-value-is-trying-to-be-set-on-a-copy-of-a-slice-from-a-d