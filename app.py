from flask import Flask, request, render_template
import pandas as pd
import os
from src.utils import load_object
from src.exception import CustomException
import sys
from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)

predict_pipeline = PredictPipeline()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Collect input data from form
            input_data = {
                'Soil Type': [request.form['soil_type']],
                'Terrain Type': [request.form['terrain_type']],
                'Crop Type': [request.form['crop_type']],
                'Water Availability': [request.form['water_availability']],
                'Irrigation System Type': [request.form['irrigation_system_type']],
                
                'Humidity (%)': [float(request.form['humidity'])],
                'Temperature (°C)': [float(request.form['temperature'])],
                'Rainfall (mm)': [float(request.form['rainfall'])],
                'Wind Speed (km/h)': [float(request.form['wind_speed'])],
                'Sunlight Hours': [float(request.form['sunlight_hours'])],
                'Soil Moisture (%)': [float(request.form['soil_moisture'])],
                'Evapotranspiration (mm/day)': [float(request.form['evapotranspiration'])],
                'Irrigation Frequency (days)': [float(request.form['irrigation_frequency'])],
                'Rainfall Forecast (mm)': [float(request.form['rainfall_forecast'])],
                'Vegetation Index': [float(request.form['vegetation_index'])],
                'Distance to Water Source (km)': [float(request.form['distance_to_water_source'])]
            }

            # Make prediction
            prediction = predict_pipeline.predict(pd.DataFrame(input_data))[0]
            return render_template('index.html', prediction=round(prediction, 2))
        except Exception as e:
            raise CustomException(e, sys)
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(host="0.0.0.0")

