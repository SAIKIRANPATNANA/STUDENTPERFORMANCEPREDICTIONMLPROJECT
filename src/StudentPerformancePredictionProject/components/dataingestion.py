import os
import sys
from src.StudentPerformancePredictionProject.exception import CustomException
from src.StudentPerformancePredictionProject.logger import logging
# from src.StudentPerformancePredictionProject.components.datatransformation import DataTransformationConfig,DataTransformation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass 
class DataIngestionConfig:
    train_data_path = os.path.join("artifacts","train.csv")
    test_data_path = os.path.join("artifacts","test.csv")
    raw_data_path = os.path.join("artifacts","data.csv")
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info("Entered into the data ingestion method")
        try:
            df = pd.read_csv('/home/user/Documents/ML DL PROJECTS/StudentPerformancePredictionMLProject/datasets/StudentsPerformance.csv')
            logging.info('Read the data as pandas dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            logging.info('Train Test Split initiated')
            df.to_csv(self.ingestion_config.raw_data_path,index=False, header=True)
            train_set,test_set = train_test_split(df,test_size=.25,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False, header=True)
            logging.info('Data Ingestion Completed')
            return (self.ingestion_config.train_data_path,self.ingestion_config.test_data_path)
        except Exception as e:
            raise CustomException(e,sys)
if __name__ == '__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()



