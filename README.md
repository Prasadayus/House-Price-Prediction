# House-Price-Prediction
This repository contains the code for a House Price Prediction application built using Streamlit. The app predicts house prices based on various input parameters provided by the user.

# Table of Contents
* Introduction
* Features
* Installation
* Usage
* Data
* Machine Learning Models
* File Descriptions
* Contributing
* License

# Introduction
The House Price Prediction app is a web application that allows users to input various parameters related to a house and predicts its price using a trained machine learning model. The app uses Gradient Boosting Regressor, which was selected after trying multiple models to achieve the best performance.

# Features
* User-friendly web interface built with Streamlit.
* Input parameters include Construction status, RERA status, BHK No. , Square Feet, Readiness to move, Resale Status, Longitude, Latitude, Posted by, Type of house (BHK or RK), and City.
* Predicts the house price based on the input parameters by using ML Model

# Installation
### 1.Clone the repository:
''' 
git clone https://github.com/Prasadayus/House-Price-Prediction.git 
'''

### 2.Create and activate a virtual environment:
python -m venv myenv
source myenv/bin/activate 

### 3.Install the required packages:
pip install -r requirements.txt

# Usage
### 1.Run the Streamlit app:
streamlit run House_pred.py

### 2.Open your web browser and go to *http://localhost:8501* to use the app.


