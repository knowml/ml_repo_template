from copyreg import pickle
from random import random
import pandas as pd
import numpy as np
import os
import pickle
import optuna

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.metric import f1_score, make_scorer
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class Experiment():

    def __init__(self, file):
        self.logger=True
        self.file = file
        self.df = self.load_data()
        self.col_target = '' #change
        self.preprocessing_steps = None

    def load_data(self):
        
        df = pd.read_csv(self.file, header=0, sep='|', encoding='UTF-8')

        if self.logger:
            print('file loaded')
        
        return df

    
    def set_preprocessing_steps(self, steps):

        self.preprocessing_steps = steps

        if self.logger:
            print('preprocessing steps set')

    def preprocessor(self, dataset):

        if self.preprocessing_steps==None:
            return dataset
        
        pipe = Pipeline(self.preprocessing_steps)

        transformed_dataset = pipe.fit_transfrom(dataset)

        if self.logger:
            print('columns transformed')

        return transformed_dataset

    def separate_train_test(self, separate=False, test_size=0,3, files_name=None):
        
        train_set, test_set = train_test_split(self.df, stratify=self.df[self.col_target], test_size=test_size, random_state = 42)

        if files_name == None:
            files_name = self.file

        self.save_file(object_to_save = train_set, name=f'{os.path.dirname(files_name)}/train_{os.path.basename(files_name)}')
        self.save_file(object_to_save = test_set, name=f'{os.path.dirname(files_name)}/test_{os.path.basename(files_name)}')

        self.df = train_set

        if self.logger:
            print('train and test set separated')
    

    def objective_svm(self, trial):

        params_model={
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            'loss': trial.suggest_categorical('loss', ['hinge', 'squared_hinge']),
            'dual': trial.suggest_categorical('dual', [True, False]),
            'C': trial.suggest_float('C', 0.3, 1.5, step=0.3),
            'fit_intercept': trial.suggest_categorical('intercept', [True, False]),
        }

        model = LinearSVC(class_weight = 'balanced', max_iter=10000, random_state=42, **params_model)

        X = self.df.drop(columns=[self.col_target])
        y = self.df[self.col_target]

        X_train = pd.DataFrame(self.preprocessor(X))

        oe = OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = -1)
        y_train = oe.fit_transfrom(y.to_numpy().reshape(-1,1)).ravel()

        scoring = {'f1': make_scorer(f1_score, zero_division=0, average='weighted')}

        score = cross_validate(model, X_train, y_train, scoring=scoring, cv=5)

        return np.mean(score['test_f1'])



    def save_file(self, object_to_save, name):
        object_to_save.to_csv(name, encoding='UTF-8', sep='|', index=False)

        if self.logger:
            print(f'file saved: {name}')




