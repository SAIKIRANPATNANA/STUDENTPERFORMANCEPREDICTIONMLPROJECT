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
from sklearn.model_selection import RandomizedSearchCV

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
            params = {
                    "RandomForest Regressor": {
                                            'n_estimators': [100, 200, 500],
                                            # 'max_features': ['auto', 'sqrt'],
                                            'max_depth': [10, 20, 30, None],
                                            'min_samples_split': [2, 5, 10],
                                            'min_samples_leaf': [1, 2, 4],
                                            'bootstrap': [True, False]
                                        },
                    "Linear Regressor": {},
                    "K-NearestNeighbor Regressor": {
                                            'n_neighbors': [3, 5, 7, 9],
                                            'weights': ['uniform', 'distance'],
                                            'metric': ['euclidean', 'manhattan']
                                        },
                    "DecisionTree Regressor": {
                                            'max_depth': [None, 10, 20, 30, 40],
                                            'min_samples_split': [2, 5, 10],
                                            'min_samples_leaf': [1, 2, 4]
                                        },
                    "SupportVectorMachine Regressor": {
                                            'C': [0.1, 1, 10, 100],
                                            'kernel': ['linear', 'rbf', 'poly'],
                                            'degree': [2, 3, 4],  # Only for 'poly' kernel
                                            'gamma': ['scale', 'auto']
                                        },
                    "AdaBoost Regressor": {
                                            'n_estimators': [50, 100, 200],
                                            'learning_rate': [0.01, 0.05, 0.1, 0.5, 1],
                                            'loss': ['linear', 'square', 'exponential']
                                        },
                    "GradientBoosting Regressor": {
                                            'n_estimators': [50, 100, 200],
                                            'learning_rate': [0.01, 0.05, 0.1],
                                            'max_depth': [3, 4, 5],
                                            'min_samples_split': [2, 5, 10],
                                            'min_samples_leaf': [1, 2, 4],
                                            'subsample': [0.8, 0.9, 1.0]
                                        },
                    "Ridge Regressor": {
                                            'alpha': [0.1, 1, 10, 100],
                                            'fit_intercept': [True, False],
                                            # 'normalize': [True, False],
                                            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
                                        },
                    "Lasso Regressor": {
                                            'alpha': [0.1, 1, 10, 100],
                                            'fit_intercept': [True, False],
                                            # 'normalize': [True, False],
                                            'precompute': [True, False],
                                            'warm_start': [True, False],
                                            'positive': [True, False]
                                        }

                    }
            model_report:dict = evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,params=params)
            best_model_score = max(list(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            if best_model_score<0.6:
                raise CustomException(f"Best model score {best_model_score} is less than 0.6, please try other models")
            logging.info(f"Best model is {best_model_name}")
            save_object(
                obj=models[best_model_name],
                filepath=self.model_trainer_config.trained_model_filepath
            )
            rs_cv = RandomizedSearchCV(models[best_model_name],param_distributions=params[best_model_name],n_iter=25)
            rs_cv.fit(x_train,y_train)
            model = models[best_model_name]
            model.set_params(**rs_cv.best_params_)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
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


