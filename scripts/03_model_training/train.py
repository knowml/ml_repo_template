import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.compose import ColumnTransformer
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt


class ModelTraining():
"""
ModelTraining class helps train final model and generate performance reports
"""

    def __init__(self, train_file, test_file, col_target, logger=True):
        self.logger=logger
        self.train_file = train_file
        self.test_file = test_file
        self.col_target = col_target
        
        self.preprocessing_steps = None

        self.train_df = self._load_data(self.train_file)
        self.test_df = self._load_data(self.test_file)
        
        self.X_train = None
        self.y_train = None
        
        self.X_test = None
        self.y_test = None
        
        
    def _logger(self, message):
        if self.logger:
            print(message)
            

    def _load_data(self, file):
        df = pd.read_csv(file, header=0, sep='|', encoding='UTF-8', dtype='float64')

        self._logger('file loaded')
        
        return df
    
    
    def remove_all_outliers(self, columns):
        
        def _get_iqr_values(col_name):
            q1 = self.train_df[col_name].quantile(0.25) 
            q3 = self.train_df[col_name].quantile(0.75) 
            iqr = q3 - q1 
            minimum  = q1-1.5*iqr 
            maximum = q3+1.5*iqr 
            return minimum, maximum
        
        for col_name in columns:
            minimum, maximum = _get_iqr_values(col_name)
            self.df = self.train_df.query(f'({col_name} > {minimum}) and ({col_name} < {maximum})')
        
        return self.df

    
    def undersample_data(self, n):
        
        df_1 = resample(self.train_df.query('target == 1.0')
                        , random_state=42
                        , n_samples=n)
        
        df_0 = resample(self.train_df.query('target == 0.0')
                        , random_state=42
                        , n_samples=n)
        
        self.df = pd.concat([df_1, df_0], axis=0).reset_index(drop=True)
        
        self._logger('undersampled data')

    
    def set_preprocessing_steps(self, steps):
        self.preprocessing_steps = steps
        
        self._logger('preprocessing steps set')
        

    def _preprocessor(self, dataset):

        if self.preprocessing_steps==None:
            return dataset
        
        pipe = ColumnTransformer(self.preprocessing_steps, remainder='passthrough').fit(dataset)

        transformed_dataset = pipe.transform(dataset)

        return transformed_dataset, pipe
    
    
    def transform_data(self):
        # transform data with preprocessing_steps
        
        
        X_train = self.train_df.drop(columns=[self.col_target])
        self.X_train , pipe = self._preprocessor(X_train)
        self.y_train = self.train_df[self.col_target].ravel()
        
        
        X_test = self.test_df.drop(columns=[self.col_target])       
        self.X_test = pipe.transform(X_test)
        self.y_test = self.test_df[self.col_target].ravel()
         
        self._logger('columns transformed')
        
    

    def train_final_model(self, model):
        
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        
        
        ## classification report
        
        class_report = classification_report(self.y_train, model.predict(self.X_train), output_dict=True, zero_division=0)
        class_report_df = pd.DataFrame(class_report).transpose().reset_index()
        self._save_file(class_report_df, 'outputs/reports/classification_report_train.csv')
        
        class_report = classification_report(self.y_test, y_pred, output_dict=True, zero_division=0)
        class_report_df = pd.DataFrame(class_report).transpose().reset_index()
        self._save_file(class_report_df, 'outputs/reports/classification_report_test.csv')

        
        # confusion matrix
        cm = confusion_matrix(self.y_test, y_pred, labels=model.classes_)
        fig = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        fig.plot()
        plt.savefig('outputs/reports/confusion_matrix.jpg')
        plt.clf()
        
        
        # feature importance
        sorted_idx = model.feature_importances_.argsort()
        columns=self.train_df.drop(columns=[self.col_target]).columns.values
        
        fig = plt.barh(width=model.feature_importances_[sorted_idx], y=columns[sorted_idx])
        plt.tight_layout()
        plt.savefig('outputs/reports/feature_importances.jpg')
        plt.clf()
    
        self._logger('model trained and reports saved')
            
    
    def _save_file(self, object_to_save, name):
        object_to_save.to_csv(name, encoding='UTF-8', sep='|', index=False)

        if self.logger:
            print(f'file saved: {name}')

            
##########################


final_training = ModelTraining(train_file = 'data/2_final/train_data.csv'
                               , test_file = f'data/2_final/test_data.csv'
                               , col_target='target')

final_training.remove_all_outliers(columns=['col1', 'col2'])

final_training.undersample_data(n=30000)

final_training.set_preprocessing_steps(steps=[
          ('oe', OrdinalEncoder(), ['cat_col'])
        , ('scaler1', StandardScaler(), ['num_col1'])
        , ('scaler2', StandardScaler(), ['num_col2'])
            ])

final_training.transform_data()

final_training.train_final_model(model = RandomForestClassifier(n_estimators=230
                                                             , criterion='entropy'
                                                             , max_depth=8
                                                             , max_features=3))