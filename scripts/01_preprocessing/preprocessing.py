import pandas as pd
import numpy as np
import os

class RawData():

    def __init__(self, file):
        self.logger = True
        self.flie = file
        self.df = self.load_data()

    def load_data(self):
        df = pd.read_csv(self.file)

        if self.logger:
            print('file loaded')

        return df

    def drop_missing(self, column=None):
        if column == None:
            self.df = self.df.dropna(axis=0).reset_index(drop=True)

            if self.logger:
                print('missings dropped based on all columns')

        else: 
            self.df = self.df.dropna(axis=0, subset=column).reset_index(drop=True)

          if self.logger:
                print(f'missings dropped based on column {column}')

    def drop_columns(self, columns):
        self.df.drop(columns=columns, inplace=True)

        if self.logger:
            print('unnecessary columns dropped')

    def add_date_feature_columns(self, date_column):

        self.df[date_column] = self.df[date_column]

        self.df['month'] = self.df[date_column].dt.month
        self.df['day_of_week'] = self.df[date_column].dt.dayofweek
        self.df['quarter'] = self.df[date_column].dt.quarter
        self.df['day_of_year'] = self.df[date_column].dt.dayofyear
        self.df['week_of_month'] = self.df[date_column].apply(lambda d: (d.day-1) // 7 +1)
        self.df['year'] = self.df[date_column].dt.year

        if self.logger:
            print('added dates features')

    
    def save_processed_file(self, name):
        self.df.to_csv(name, encoding='UTF-8', sep='|', index=False)
        
        if self.logger:
            print('file saved')




