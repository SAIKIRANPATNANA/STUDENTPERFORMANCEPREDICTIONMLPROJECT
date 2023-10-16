import os
import sys
import pickle as pkl
from src.StudentPerformancePredictionProject.exception import CustomException
def save_object(obj, file_path):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path, 'wb') as f:
            pkl.dump(obj, f)
    except Exception as e:
        raise CustomExceptionl(e,sys)