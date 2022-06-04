import pandas as pd
import numpy as np
import os
import optuna

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import recall_score, make_scorer
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


class Experiment():
    """
    Experiment class transforms processed data, seprates for training and test, runs experiments with optuna for diferent classifiers
    """

    def __init__(self, file, col_target, logger=True):
        self.logger=logger
        self.file = file
        self.df = self._load_data()
        self.col_target = col_target
        
        self.preprocessing_steps = None
        self.X_train = None
        self.y_train = None
        
        
    def _logger(self, message):
        if self.logger:
            print(message)
            

    def _load_data(self):
        df = pd.read_csv(self.file, header=0, sep='|', encoding='UTF-8', dtype='float64')

        self._logger('file loaded')
        
        return df
    
    def separate_train_test(self, separate=False, test_size=0.3, files_name=None):
        # if separate=True: separate loaded file into two files for:
        # 1/ expriments (this script)
        # 2/ optimisation and test
        # if separate=False: use data set from loaded file for experiments, optimisation and test on the same data
        
        train_set, test_set = train_test_split(self.df, stratify=self.df[self.col_target], test_size=test_size, random_state = 42)

        if files_name == None:
            files_name = self.file

        self._save_file(object_to_save = train_set, name=f'{os.path.dirname(files_name)}/train_{os.path.basename(files_name)}')
        self._save_file(object_to_save = test_set, name=f'{os.path.dirname(files_name)}/test_{os.path.basename(files_name)}')

        self.df = train_set

        self._logger('train and test set separated')
    
    
    def remove_all_outliers(self, columns):
        
        def _get_iqr_values(col_name):
            q1 = self.df[col_name].quantile(0.25) 
            q3 = self.df[col_name].quantile(0.75) 
            iqr = q3-q1 #Interquartile range
            minimum  = q1-1.5*iqr 
            maximum = q3+1.5*iqr 

            return minimum, maximum
        
        for col_name in columns:
        
            minimum, maximum = _get_iqr_values(col_name)
            self.df = self.df.query(f'({col_name} > {minimum}) and ({col_name} < {maximum})')
        
        return self.df

    
    def undersample_data(self, n):
        
        df_1 = resample(self.df.query('target == 1.0')
                        , random_state=42, n_samples=n)
        
        df_0 = resample(self.df.query('target == 0.0')
                        , random_state=42, n_samples=n)
        
        self.df = pd.concat([df_1, df_0], axis=0).reset_index(drop=True)
        
        self._logger('undersampled data')

    
    def set_preprocessing_steps(self, steps):
        self.preprocessing_steps = steps
        
        self._logger('preprocessing steps set')
           
    
    def transform_data(self):
        # transform data with preprocessing_steps
        
        X = self.df.drop(columns=[self.col_target])
        y = self.df[self.col_target]
        
        self.X_train = self._preprocessor(X)
        self.y_train = y.ravel()
         
        self._logger('columns transformed')
        
        
    def _preprocessor(self, dataset):
        
        if self.preprocessing_steps==None:
            return dataset
        
        pipe = ColumnTransformer(self.preprocessing_steps, remainder='passthrough')
        transformed_dataset = pipe.fit_transform(dataset)

        return transformed_dataset
        
        
    def call_optuna_study(self, objective, n_trials=10):
        
        study = optuna.create_study(direction='maximize')
        
        if objective=='svm':
            obj = self._objective_svm
        elif objective=='randforest':
            obj = self._objective_randforest
        elif objective=='knn':
            obj = self._objective_knn
        else:
            self._logger(f'no objective settings for {objective}')
            
        study.optimize(obj, n_trials=n_trials)
        df = study.trials_dataframe(attrs={'number', 'value', 'params', 'state'})
        self._save_file(df, f'scripts/02_model_selection/{objective}_trails_results.csv')
        
        self._logger(f'study for {objective} finished')
    
    
    def _objective_svm(self, trial):
        
        params_model={
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            'loss': trial.suggest_categorical('loss', ['hinge', 'squared_hinge']),
            'dual': trial.suggest_categorical('dual', [True, False]),
            'C': trial.suggest_float('C', 0.3, 1.5, step=0.3),
            'fit_intercept': trial.suggest_categorical('intercept', [True, False]),
        }
        model = LinearSVC(class_weight = 'balanced', max_iter=5000, random_state=42, **params_model)
        
        return self._objective(trial, model)
    

    def _objective_knn(self, trial):
        
        params_model={    
            'n_neighbors': trial.suggest_int('n_neighbors', 200, 500, step=10),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'algorithm': trial.suggest_categorical('alo', ['auto', 'kd_tree']),
            'p': trial.suggest_int('p', 1, 2, step=1)
        }
        model = KNeighborsClassifier(**params_model)
        
        return self._objective(trial, model)

            
    def _objective_randforest(self, trial):

        params_model={
            'n_estimators': trial.suggest_int('estimators', 10, 290, step=20),
            'criterion': trial.suggest_categorical('crit', ['gini', 'entropy']),
            'max_depth': trial.suggest_int('max_depth', 2, 10, step=2),
            'max_features': trial.suggest_int('max_features', 3, 8, step=1)
        }
        model = RandomForestClassifier(random_state=42, **params_model)
        
        return self._objective(trial, model)
    
    
    def _objective(self, trial, model):
        
        try:
            model.fit(self.X_train, self.y_train)
        except ValueError:
            print('Unsupported set of arguments')
            return 0
        
        scoring = {'recall': make_scorer(recall_score, zero_division=0, average='weighted')}
            
        score = cross_validate(model, self.X_train, self.y_train, scoring=scoring, cv=5)

        return np.mean(score['test_recall'])

    
    def _save_file(self, object_to_save, name):
        object_to_save.to_csv(name, encoding='UTF-8', sep='|', index=False)

        self._logger(f'file saved: {name}')

            
##########################

# run experiments to select the best classifier

experiment = Experiment(file = 'data/2_final/data.csv', col_target='target')

# separate file into train/validation and test data
experiment.separate_train_test(separate=True)

experiment.remove_all_outliers(columns=['col1', 'col2'])

# undersample data, sample n observation from every class
experiment.undersample_data(n=10000)


# define transformation of data
experiment.set_preprocessing_steps(steps=[
          ('oe', OrdinalEncoder(), ['cat_col'])
        , ('scaler1', StandardScaler(), ['num_col1'])
        , ('scaler2', StandardScaler(), ['num_col2'])
            ])

# apply transformation
experiment.transform_data()

# run optuna studies to find optimised hyperparameters
n_trials=50

experiment.call_optuna_study('svm', n_trials=n_trials)
experiment.call_optuna_study('randforest', n_trials=n_trials)
experiment.call_optuna_study('knn', n_trials=n_trials)