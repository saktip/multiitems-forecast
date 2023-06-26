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

        for k in range(len(self.list_items)):
            df_temp = self.df[(self.df[self.item_column_name] == self.list_items[k])].copy()
            df_temp['cap'] = 50
            df_temp['floor'] = 0
            temp_mae[self.list_items[k]] = []
            print(self.list_items[k])
            for params in all_params:
                #fb prophet "will by default fit weekly and yearly seasonalities"
                m = Prophet(**params, growth='logistic').fit(df_temp) 
                df_cv = cross_validation(m, initial='7 days', horizon='30 days', parallel="processes")
                df_p = performance_metrics(df_cv, rolling_window=0) #kalo muncul assertion error untuk growth='logistic', rolling window isi 0 
                temp_mae[self.list_items[k]].append(df_p['mae'].values[0])
                maes[self.list_items[k]] = temp_mae[self.list_items[k]]
                
            self.tuning_results[self.list_items[k]] = pd.DataFrame(all_params)
            self.tuning_results[self.list_items[k]]['mae'] = pd.DataFrame(maes[self.list_items[k]])
            self.tuning_results[self.list_items[k]] = self.tuning_results[self.list_items[k]].sort_values(['mae'])
        return self.df, self.list_items, self.tuning_results, self.item_column_name

class ForecastWithParam():
    
    df = pd.DataFrame()
    list_items = []
    tuning_results = {}
    dfs = {}
    item_column_name = ''
    
    def __init__(self, tuning_instance):
        self.df = tuning_instance.df
        self.list_items = tuning_instance.list_items
        self.tuning_results = tuning_instance.tuning_results
        self.item_column_name = tuning_instance.item_column_name

    def proccess_forecastwithparam(self, 
                                   periods, 
                                   yearly_seasonality = 'auto', 
                                   weekly_seasonality = 'auto', 
                                   daily_seasonality='auto',
                                   growth={'type':'linear', 'cap':None, 'floor':None}):
        forecast_all = {}

        for i in range(len(self.list_items)): 
            # harus ada kolom ds, y, cap, dan floor untuk forecast dengan fb logistic
            self.dfs[self.list_items[i]] = self.df[['ds', 'y']][(self.df[self.item_column_name] == self.list_items[i])].assign(cap=50).assign(floor=0).reset_index(drop=True)

        for name, df in self.dfs.items():
            future_fit = Prophet(changepoint_prior_scale=0.75, seasonality_prior_scale = 0.5,).fit(self.dfs[list(self.dfs)[0]]) # x ini hanya sekadar ngebuat tanggal predict sebanyak periods
            future_df = future_fit.make_future_dataframe(periods=periods)
            if (growth['type'] == 'logistic'):
                future_df['cap'] = growth['cap']
                future_df['floor'] = growth['floor']
            forecast_future_df = pd.merge(df[['ds', 'y']], future_df, on='ds', how='outer')
            # if name ini or ini or ini (regex?)
            m = Prophet(changepoint_prior_scale=self.tuning_results[name]['changepoint_prior_scale'].iloc[0], 
                        seasonality_prior_scale=self.tuning_results[name]['seasonality_prior_scale'].iloc[0],
                        yearly_seasonality=yearly_seasonality,
                        weekly_seasonality=weekly_seasonality,
                        daily_seasonality=daily_seasonality,
                        growth=growth['type']).fit(df)
            forecast = m.predict(future_df)
            forecast['yhat_' + name] = forecast['yhat'] # ngebuat kolom baru dengan nama kolom yhat_kode dan isi berupa yhat
            forecast_all[name] = pd.merge(forecast_future_df, forecast[['ds', 'yhat_' + name]], on='ds', how='left').drop_duplicates(subset=['ds']).reset_index(drop=True) 
            # print(pd.merge(forecast_future_df, forecast[['ds', 'yhat_' + name]], on='ds', how='left').drop_duplicates(subset=['ds']).reset_index(drop=True))
            # type: ignore # drop duplicate hasil merge via loop
            # kalo gak reset index, nanti index-nya gak urut 1,2,3, dst karena sebelumnya di-drop duplicate
        return forecast_all


# https://stackoverflow.com/questions/64961448/python-pandas-warning-a-value-is-trying-to-be-set-on-a-copy-of-a-slice-from-a-d