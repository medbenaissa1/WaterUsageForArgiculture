import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")
    
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        
        '''
            Method for data transformation

        '''
        try:
            categorical_columns = [
    'Soil Type', 'Terrain Type', 'Crop Type', 'Water Availability', 
    'Irrigation System Type'
]
            
            
            numerical_columns =  ['Humidity (%)', 'Temperature (°C)', 'Rainfall (mm)', 'Wind Speed (km/h)', 'Sunlight Hours', 'Soil Moisture (%)', 'Evapotranspiration (mm/day)', 'Irrigation Frequency (days)', 'Rainfall Forecast (mm)', 'Vegetation Index', 'Distance to Water Source (km)']
            num_pipeline = Pipeline(
                
                steps = [
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler(with_mean=False))
                     
                ]
                
            
            )
            logging.info("Numerical Columns Standard Scaling Done")

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info("Categorical Columns Encoding Done")
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )
            logging.info("preprocessiong.........")
            
            return preprocessor
        
    
   
            
            
            
        except Exception as e :
            raise CustomException(e,sys) 
        
        
        
    
    
    def initiate_data_transformation(self,train_path,test_path,val_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            val_df = pd.read_csv(val_path)
            
            logging.info("read test and train data...")
            logging.info("Get Preprocessing Object")
            
            preprocessing_object = self.get_data_transformer_object()
            
            target_column_name = "Daily Irrigation Water (m³)"
            
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            input_feature_val_df = val_df.drop(columns=[target_column_name],axis=1)
            target_feature_val_df = val_df[target_column_name]
            
            logging.info("Applying preprocessing object on test and train dataframe....")
            input_feature_train_arr = preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_object.transform(input_feature_test_df)
            input_feature_val_arr = preprocessing_object.transform(input_feature_val_df)
            
            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            
            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]
            
            val_arr = np.c_[
                input_feature_val_arr,np.array(target_feature_val_df)
            ]
            
            logging.info("Saving preprocessing object")
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_object
            )
            return (
                train_arr , 
                test_arr,
                val_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

            
            
            
            
            
            
        except Exception as e :
            raise CustomException(e,sys)
            