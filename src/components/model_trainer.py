import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utils import evaluate_model,save_object

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig:
    training_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Split training and test input data')
            X_train,X_test,Y_train,Y_test=(
                train_array[:,:-1],
                test_array[:,:-1],
                train_array[:,-1],
                test_array[:,-1]
            )
            models={
                'RandomForest':RandomForestRegressor(),
                'Decision Tree':DecisionTreeRegressor(),
                'Linear Regression':LinearRegression(),
                'AdaBoostRegressor':AdaBoostRegressor(),
                'Gradient Booster':GradientBoostingRegressor(),
                'KNR':KNeighborsRegressor(),
                'CatBoost':CatBoostRegressor(),
                'XgBoost':XGBRegressor(),
                'SVR':SVR()
            }
            params={
                'Linear Regression':{
                    'n_jobs':[2,3,4,5]
                },
                'RandomForest':{
                    'n_estimators':[100,200,300,500,1000],
                    'criterion':['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                    'max_depth':[2,3,4,5,6],
                    'max_features':['sqrt','log2',5,10,2,3]
                },
                'Decision Tree':{
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter': ['best', 'random'],
                    'max_depth':[100,200,300,500,1000],
                    'max_features':['auto', 'sqrt', 'log2']
                },
                'AdaBoostRegressor':{
                    'n_estimators':[100,200,300,50,500],
                    'loss': ['linear', 'square', 'exponential']
                },
                'Gradient Booster':{
                    'loss':['squared_error', 'absolute_error', 'huber', 'quantile'],
                    'n_estimators': [100,200,300,500,1000],
                    'criterion': ['friedman_mse', 'squared_error'],
                    'max_depth':[2,3,4,5,6,7]
                },
                'KNR':{
                    'n_neighbors': [5,7,9,11,3],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                },
                'CatBoost':{
                    'iterations':[100,30,40,50],      
                    'learning_rate':[0.1,0.01,0.001],   
                    'depth':[2,3,4,5,6]              
                },
                'XgBoost':{
                    'max_depth':[2,3,5,4],
                    'learning_rate':[0.1,0.01,0.001],
                    'n_estimators':[100,200,300,500],
                },
                'SVR':{
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                    'gamma': ['scale', 'auto',0.1,1],
                    'C': [0.1,1,10],
                    'epsilon':[0.1,0.2,0.3,0.05,0.01]
                }
            }
            model_report:dict=evaluate_model(X_train,Y_train,X_test,Y_test,models,params)

            # to get the best model with best score
            best_model_name=''
            best_score=0
            for i,j in model_report.items():
                if j>best_score:
                    best_score=j
                    best_model_name=i
            best_model=models[best_model_name]
            if best_score<0.6:
                raise CustomException("No best model found",sys)
            logging.info('Best found model on both training and testing dataset')

            save_object(
                file_path=self.model_trainer_config.training_model_file_path,
                obj=best_model
            )

            prediction=best_model.predict(X_test)
            r2_square=r2_score(Y_test,prediction)
            return r2_square
        except Exception as e:
            raise CustomException(e,sys)