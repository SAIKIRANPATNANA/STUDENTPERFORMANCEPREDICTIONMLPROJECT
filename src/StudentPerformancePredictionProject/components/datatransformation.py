import sys
import os
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from src.StudentPerformancePredictionProject.exception import CustomException
from src.StudentPerformancePredictionProject.logger import logging
from src.StudentPerformancePredictionProject.utils.common import save_object
from src.StudentPerformancePredictionProject.components.dataingestion import DataIngestion
@dataclass
class DataTransformationConfig:
    preprocessor_obj_filepath = os.path.join('artifacts','preprocessor.pkl')
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def get_data_transformer_object(self):
        try:
            num_features = ['reading score', 'writing score']
            cat_features = ['gender','race/ethnicity','parental level of education','lunch','test preparation course']
            num_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),('scaler',StandardScaler())])
            cat_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),('ohe',OneHotEncoder())])
            logging.info('Numerical pipeline and categorical pipeline are implemented')
            preprocessor = ColumnTransformer(transformers=[('num_pipeline', num_pipeline, num_features),('cat_pipeline', cat_pipeline, cat_features)])
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_data_path, test_data_path):
        try:
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)
            logging.info("Reading Train Data and Test Data completed")
            logging.info("Getting Preprcessing Object")
            preprocessor = self.get_data_transformer_object()
            target_col = 'math score'
            input_features_train = train_data.drop(target_col,axis=1)
            target_feature_train = train_data[target_col]
            input_features_test = test_data.drop(target_col,axis=1)
            target_feature_test = test_data[target_col]
            input_features_train_array = preprocessor.fit_transform(input_features_train)
            save_object(
                obj = preprocessor,
                file_path = self.data_transformation_config.preprocessor_obj_filepath
            )
            logging.info("Preprocessing object is saved")
            input_features_test_array = preprocessor.transform(input_features_test)
            train_array = np.c_[(input_features_train_array,np.array(target_feature_train))]
            test_array = np.c_[(input_features_test_array,np.array(target_feature_test))]
            return (train_array, test_array, self.data_transformation_config.preprocessor_obj_filepath)
        except Exception as e:
            raise CustomException(e,sys)

if __name__=='__main__':
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()
    obj = DataTransformation()
    obj.initiate_data_transformation(train_data,test_data)



