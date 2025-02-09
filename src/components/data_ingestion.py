from src.logger import logging
import pandas as pd
import os
import sys
from src.exception import CustomException

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_predictor import ModelPredictor

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', "raw.csv")
    train_data_path: str = os.path.join('artifacts', "train.csv")
    val_data_path: str = os.path.join('artifacts', "val.csv")  # Nouveau fichier pour validation
    test_data_path: str = os.path.join('artifacts', "test.csv")

class DataIngestion:
    def __init__(self, validation_ratio=0.2, test_ratio=0.2):
        self.ingestion_config = DataIngestionConfig()
        self.validation_ratio = validation_ratio  # 20% pour validation
        self.test_ratio = test_ratio  # 20% pour test

    def initiate_data_ingestion(self):
        logging.info("Début de l'ingestion des données")
        try:
            df = pd.read_csv("C:/Users/E15/Desktop/Workshop Project/notebook/data/morocco_irrigation_dataset.csv")
            df.drop(columns = ["Recommended Irrigation Level"],axis=1)
            
            logging.info("Lecture du dataset réussie")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Division Train/Test")
            train_set, test_set = train_test_split(df, test_size=self.test_ratio, random_state=44)

            logging.info("Division Train/Validation")
            train_set, val_set = train_test_split(train_set, test_size=self.validation_ratio, random_state=44)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            val_set.to_csv(self.ingestion_config.val_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion des données et séparation terminée")

            return (
                self.ingestion_config.train_data_path,
                
                self.ingestion_config.test_data_path,
                self.ingestion_config.val_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


        
        
    
if __name__ == "__main__":
    
    objet = DataIngestion()
    train_data, test_data, val_data = objet.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr, test_arr, val_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data, val_data)
    
    #model_trainer = ModelTrainer()
    #score = model_trainer.initiate_model_trainer(train_arr, test_arr, val_arr)
    #print("Model R2 Score:", score)
    

    # Example input data
    
    input_data = {
        "Humidity (%)": [45],
        "Temperature (°C)": [28],
        "Rainfall (mm)": [5], 
        "Wind Speed (km/h)": [10],
        "Sunlight Hours": [8],
        "Soil Moisture (%)": [30],
        "Evapotranspiration (mm/day)": [4.5], #calculated
        "Irrigation Frequency (days)": [3],
        "Rainfall Forecast (mm)": [7], 
        "Vegetation Index": [0.65], #calculated
        "Distance to Water Source (km)": [1.5], 
        "Soil Type": ["Clay"],
        "Terrain Type": ["Mountain"],
        "Crop Type": ["Wheat"],
        "Water Availability": ["High"],
        "Irrigation System Type": ["Drip"],
        
    }
    
    

    # Initialize predictor
    predictor = ModelPredictor()

    # Make prediction
    predicted_value = predictor.predict(input_data)
    print(f"🔹 Predicted Daily Irrigation Water (m³): {predicted_value}")


        
        
        
        
        

