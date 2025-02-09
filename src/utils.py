import os
import sys
import dill
import pickle
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """Sauvegarde un objet en utilisant Dill."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """Charge un objet enregistré avec Pickle."""
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


    