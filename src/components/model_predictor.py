import os
import pandas as pd
import numpy as np
from src.utils import load_object
from src.exception import CustomException
import sys

class ModelPredictor:
    def __init__(self):
        try:
            self.model_path = os.path.join("artifacts", "model.pkl")
            self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            # Load trained model and preprocessor
            self.model = load_object(self.model_path)
            self.preprocessor = load_object(self.preprocessor_path)
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, input_data):
        """
        Predicts Daily Irrigation Water (m³) based on input data.

        Args:
            input_data (dict): Dictionary containing feature values.

        Returns:
            float: Predicted value.
        """
        try:
            # Convert input dictionary to DataFrame
            input_df = pd.DataFrame(input_data)

            # Apply preprocessing (transform only, no fit)
            processed_data = self.preprocessor.transform(input_df)

            # Make predictions
            prediction = self.model.predict(processed_data)

            return prediction[0]

        except Exception as e:
            raise CustomException(e, sys)
