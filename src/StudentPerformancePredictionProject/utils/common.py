import os
import sys
import pickle as pkl
from src.StudentPerformancePredictionProject.exception import CustomException
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score
import warnings as warn
warn.filterwarnings('ignore')
def save_object(obj, filepath):
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path,exist_ok=True)
        with open(filepath, 'wb') as f:
            pkl.dump(obj, f)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(x_train,y_train,x_test,y_test,models,params):
    try:
        model_report = {}
        for model_name,model in models.items():
            rs_cv = RandomizedSearchCV(model,param_distributions=params[model_name],n_iter=25)
            rs_cv.fit(x_train,y_train)
            model.set_params(**rs_cv.best_params_)
            model.fit(x_train,y_train)
            y_pred = model.predict(x_test)
            model_report[model_name] = r2_score(y_test,y_pred)
        return model_report
    except Exception as e:
        raise CustomException(e,sys)

def load_object(filepath):
    try:
        with open(filepath, 'rb') as f:
            return pkl.load(f)
    except Exception as e:
        raise CustomException(e,sys)