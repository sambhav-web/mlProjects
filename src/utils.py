import os 
import sys
import dill
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path:str,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file:
            dill.dump(obj,file)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(x_train,y_train,x_test,y_test,models,params):
    try:
        report={}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            para=params[list(models)[i]]
            gs=GridSearchCV(model,param_grid=para,cv=3)
            gs.fit(x_train,y_train)
            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)
            y_train_pred=model.predict(x_train)
            y_test_pred=model.predict(x_test)
            y_train_score=r2_score(y_train,y_train_pred)
            y_test_score=r2_score(y_test,y_test_pred)
            report[list(models.keys())[i]]=y_test_score
            logging.info(f"{list(models)[i]} is trained")
        return report
    except Exception as e:
        raise CustomException(e,sys)