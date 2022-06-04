import pandas as pd
import numpy as np
import os

class RawData():
"""
RawData class helps to generate processed file for modeling
"""

    def __init__(self, file, logger=True):
        self.logger = logger
        self.file = file
        self.df = self._load_data()
            
    def _logger(self, message):
        if self.logger:
            print(message)

    def _load_data(self):
        df = pd.read_csv(self.file
                   , sep=','
                   , header=0
                   , parse_dates=['date_column']
                   , infer_datetime_format=True
                   , low_memory=False)
        
        self._logger('file loaded')

        return df
    
    def map_categorical_values(self, new_cl, old_col, map_dict):
        self.df[new_cl] = self.df[old_col].map(map_dict)
        self.df.drop(columns=old_col, inplace=True)
        
        self._logger(f'{old_col} mapped')
        
    def subset_df(self, query):
        self.df = self.df.query(query).reset_index(drop=True)
        
        self._logger(f'subset to {query}')
    

    def drop_missing(self, column=None):
        if column == None:
            self.df = self.df.dropna(axis=0).reset_index(drop=True)
            
            self._logger('missings dropped based on all columns')
            
        else: 
            self.df = self.df.dropna(axis=0, subset=column).reset_index(drop=True)
            
            self._logger(f'missings dropped based on column {column}')
            

    def drop_columns(self, columns):
        self.df.drop(columns=columns, inplace=True)

        self._logger('unnecessary columns dropped')
        
    def encode_dummies(self, column):
        self.df = pd.get_dummies(self.df, columns=column, drop_first=True)
        
        self._logger(f'dummies ready for {column}')

    def add_date_feature_columns(self, date_column):
        self.df[f'{date_column}_year'] = self.df[date_column].dt.year
        self.df[f'{date_column}_month'] = self.df[date_column].dt.month
        self.df.drop(columns=date_column, inplace=True)
        
        self._logger('added dates features')
        
    def log_transformation(self, column):
        self.df[column] = self.df[column].apply(lambda x: np.log(x+0.01))
        
        self._logger(f'log transformation on {column}')

    
    def save_processed_file(self, name):
        self.df.to_csv(name, encoding='UTF-8', sep='|', index=False)
        
        self._logger('file saved')



#######

file = 'data/0_raw/data.csv'

df = RawData(file=file)


target_map = {"cat 0": 0, "cat 1":1, "cat irrelevant": -1}
df.map_categorical_values(new_cl='target', old_col='old_name', map_dict=target_map)
df.subset_df('target >= 0')


# 3. 
df.drop_columns(columns=['co1', 'col2'])
    
# 4.                        
df.drop_missing(column=['col3'])
df.drop_missing(column=['col4'])
 
# 5.
col_map = {'val1': 0, ' val2': 1, 'vl3': 1}
df.map_categorical_values(new_cl='new_col', old_col='old_col', map_dict=col_map)

df.encode_dummies(column=['col5'])
                         
# 5.                     
df.add_date_feature_columns(date_column='date_col')
df.log_transformation(column='skewed_col')


df.save_processed_file('data/2_final/data.csv')
