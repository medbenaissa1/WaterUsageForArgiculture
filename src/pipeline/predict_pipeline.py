import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os 

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        self.model = load_object(self.model_path)
        self.preprocessor = load_object(self.preprocessor_path)

    def predict(self, features):
        try:
            data_scaled = self.preprocessor.transform(features)
            preds = self.model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, soil_type, terrain_type, crop_type, water_availability,
                 irrigation_system_type, humidity, temperature, rainfall,
                 wind_speed, sunlight_hours, soil_moisture, evapotranspiration, irrigation_frequency,
                 rainfall_forecast, vegetation_index, distance_to_water_source):

        self.soil_type = soil_type
        self.terrain_type = terrain_type
        self.crop_type = crop_type
        self.water_availability = water_availability
        self.irrigation_system_type = irrigation_system_type
        
        self.humidity = humidity
        self.temperature = temperature
        self.rainfall = rainfall
        self.wind_speed = wind_speed
        self.sunlight_hours = sunlight_hours
        self.soil_moisture = soil_moisture
        self.evapotranspiration = evapotranspiration
        self.irrigation_frequency = irrigation_frequency
        self.rainfall_forecast = rainfall_forecast
        self.vegetation_index = vegetation_index
        self.distance_to_water_source = distance_to_water_source

    def get_data_as_data_frame(self):
        try:
            # Create a dictionary with all required columns and their values
            data_dict = {
                'Soil Type': [self.soil_type],
                'Terrain Type': [self.terrain_type],
                'Crop Type': [self.crop_type],
                'Water Availability': [self.water_availability],
                'Irrigation System Type': [self.irrigation_system_type],
                
                'Humidity (%)': [self.humidity],
                'Temperature (°C)': [self.temperature],
                'Rainfall (mm)': [self.rainfall],
                'Wind Speed (km/h)': [self.wind_speed],
                'Sunlight Hours': [self.sunlight_hours],
                'Soil Moisture (%)': [self.soil_moisture],
                'Evapotranspiration (mm/day)': [self.evapotranspiration],
                'Irrigation Frequency (days)': [self.irrigation_frequency],
                'Rainfall Forecast (mm)': [self.rainfall_forecast],
                'Vegetation Index': [self.vegetation_index],
                'Distance to Water Source (km)': [self.distance_to_water_source]
            }
            
            # Ensure DataFrame columns match the preprocessor expectations
            return pd.DataFrame(data_dict)
        except Exception as e:
            raise CustomException(e, sys)