import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.best_model = None

    def initiate_model_trainer(self, train_array, test_array, val_array):
        try:
            logging.info("Splitting training and test input data")
            
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            X_val, y_val = val_array[:, :-1], val_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            param_grid = {
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                },
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse'],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128],
                    'max_depth': [3, 5, 7, 9],
                    'min_samples_split': [2, 5, 10],
                },
                "Linear Regression": {},
                "CatBoost Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100],
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128],
                },
            }

            model_report = self.evaluate_models_with_grid_search(X_train, y_train, X_val, y_val, models, param_grid)
            
            if not model_report:
                raise CustomException("No models were successfully trained.")
            
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No suitable model found with an R2 score above 0.6")

            self.best_model = models[best_model_name]
            logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score}")
            
            # Entraînement du meilleur modèle sur l'ensemble d'entraînement complet
            logging.info("Training the best model on the full training set")
            self.best_model.fit(X_train, y_train)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=self.best_model,
            )

            predicted = self.best_model.predict(X_test)
            return r2_score(y_test, predicted)

        except Exception as e:
            logging.error(f"Error in initiate_model_trainer: {str(e)}")
            raise CustomException(e, sys)

    def evaluate_models_with_grid_search(self, X_train, y_train, X_test, y_test, models, param_grid):
        try:
            model_scores = {}

            for name, model in models.items():
                logging.info(f"🔍 Performing GridSearchCV for: {name}")
                
                params = param_grid.get(name, {})
                if params:
                    grid_search = GridSearchCV(
                        estimator=model,
                        param_grid=params,
                        cv=3,
                        n_jobs=-1,
                        scoring="r2",
                        verbose=1,
                    )
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                else:
                    best_model = model
                    best_model.fit(X_train, y_train)
                
                cv_scores = cross_val_score(best_model, X_train, y_train, cv=3, scoring="r2")
                logging.info(f"Cross-validation R2 scores for {name}: {cv_scores}")

                y_test_pred = best_model.predict(X_test)
                r2 = r2_score(y_test, y_test_pred)
                model_scores[name] = r2
                logging.info(f"R2 score for {name}: {r2}")

            return model_scores

        except Exception as e:
            logging.error(f"Error in evaluate_models_with_grid_search: {str(e)}")
            raise CustomException(e, sys)

    def predict(self, input_data):
        if self.best_model is None:
            raise CustomException("No trained model available. Please train the model first.")
        return self.best_model.predict(input_data)
