from dataclasses import dataclass
import os
import sys
import pandas as pd
import numpy as np

from src.utils import save_object
from src.logger import logging
from src.exception import CustomException
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

@dataclass
class DataTransformerConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformer:
    def __init__(self):
        self.data_transformation_config=DataTransformerConfig()
    def get_transformer_object(self):
        """To create Data Transformer Object"""
        try:
            numerical_columns=["reading_score","writing_score"]
            categorical_columns=["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]
            
            numerical_pipeline=Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
            ])
            categorical_pipeline=Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('encoder',OneHotEncoder(drop='first',sparse_output=False)),
                ('scaler',StandardScaler())
            ])
            logging.info('Categorical Columns Encoding Completed')
            logging.info('Numerical Columns Scaling Completed')

            preprocessor=ColumnTransformer(transformers=[
                ('numerical_data',numerical_pipeline,numerical_columns),
                ('categorical_data',categorical_pipeline,categorical_columns)
            ])
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformer(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info('Read the train test Dataset')
            target_column='math_score'
            preprocessor_obj=self.get_transformer_object()

            input_feature_train_df=train_df.drop(columns=['math_score'])
            target_feature_train_df=train_df[target_column]        
            input_feature_test_df=test_df.drop(columns=['math_score'])
            target_feature_test_df=test_df[target_column]   
            logging.info('Applying Preprocessing object on training and testing dataframes')     

            input_feature_train_array=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_array=preprocessor_obj.transform(input_feature_test_df)

            train_array=np.c_[input_feature_train_array,np.array(target_feature_train_df)]
            test_array=np.c_[input_feature_test_array,np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            logging.info('Saved Preprocessing object')

            return(
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)