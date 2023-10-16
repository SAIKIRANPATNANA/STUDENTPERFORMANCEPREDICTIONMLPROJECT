import os
import sys
from dataclasses import dataclass
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression,Ridge,Lasso 
from sklearn.model_selection import RandomizedSearchCV
from src.StudentPerformancePredictionProject.exception import CustomException
from src.StudentPerformancePredictionProject.logger import logging
from src.StudentPerformancePredictionProject.utils.common import save_object
from src.StudentPerformancePredictionProject.utils.common import evaluate_model
from src.StudentPerformancePredictionProject.components.dataingestion import DataIngestion
from src.StudentPerformancePredictionProject.components.datatransformation import DataTransformation

@dataclass
class ModelTrainingConfig:
    trained_model_filepath = os.path.join('artifacts','model.pkl')

class ModelTraining:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()
    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("Splitting train and test data")
            x_train,y_train,x_test,y_test = (train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])
            models = {"RandomForest Regressor": RandomForestRegressor(),
                    "Linear Regressor": LinearRegression(),
                    "K-NearestNeighbor Regressor": KNeighborsRegressor(), 
                    "DecisionTree Regressor": DecisionTreeRegressor(),
                    "SupportVectorMachine Regressor": SVR(),
                    "AdaBoost Regressor": AdaBoostRegressor(),
                    "GradientBoosting Regressor": GradientBoostingRegressor(),
                    "Ridge Regressor": Ridge(),
                    "Lasso Regressor": Lasso()}
            model_report:dict = evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)
            best_model_score = max(list(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            if best_model_score<0.6:
                raise CustomException(f"Best model score {best_model_score} is less than 0.6, please try other models")
            logging.info(f"Best model is {best_model_name}")
            save_object(
                obj=models[best_model_name],
                filepath=self.model_trainer_config.trained_model_filepath
            )
            models[best_model_name].fit(x_train, y_train)
            y_pred = models[best_model_name].predict(x_test)
            return r2_score(y_test, y_pred)
        except Exception as e:
            logging.info(f"Exception occurred while initiating model training: {e}")
            raise CustomException(e,sys)
        
        
if __name__=='__main__':
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()
    obj = DataTransformation() 
    train_array,test_array =  obj.initiate_data_transformation(train_data,test_data)
    obj = ModelTraining()
    r2_score = obj.initiate_model_training(train_array,test_array)
    logging.info(f"R2 score is {r2_score}")


