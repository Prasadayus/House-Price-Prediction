# HOURICEP -The House-Price-Prediction APP
This repository contains the code for a House Price Prediction application built using Streamlit. The app predicts house prices based on various input parameters provided by the user.
![Screenshot 2024-06-15 000322](https://github.com/Prasadayus/House-Price-Prediction/assets/129419372/c177a21f-d7b7-4739-89ae-bd44ef46dc9c)



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
```
git clone https://github.com/Prasadayus/House-Price-Prediction.git 
```

### 2.Create and activate a virtual environment:
```
python -m venv myenv
source myenv/bin/activate
```

### 3.Install the required packages:
```
pip install -r requirements.txt
```

# Usage
### 1.Run the Streamlit app:
```
streamlit run House_pred.py
```
### 2.Open your web browser and go to *http://localhost:8501* to use the app.

# Data
The dataset is 'House Price Prediction Challenge.csv' file in this repo
The dataset used for training the model contains the following columns:

* POSTED_BY: The person who posted the listing (Owner, Dealer, Builder).
* UNDER_CONSTRUCTION: Whether the house is under construction (0 or 1).
* RERA: RERA status (0 or 1).
* BHK_NO.: Number of bedrooms.
* BHK_OR_RK: Type of house (BHK or RK).
* SQUARE_FT: Square footage of the house.
* READY_TO_MOVE: Whether the house is ready to move in (0 or 1).
* RESALE: Whether the house is a resale property (0 or 1).
* ADDRESS: Address of the house.
* LONGITUDE: Longitude of the house location.
* LATITUDE: Latitude of the house location.
* TARGET(PRICE_IN_LACS): Price of the house in lakhs.

# File Descriptions

* `House_pred.py`: Main script to run the Streamlit app.
* `House_pred.ipynb`: Jupyter notebook for machine learning model training and evaluation.
* `label_house_encoder.pkl`: Pickle file containing the LabelEncoder for categorical variables.
* `gbr_house_model.pkl`: Pickle file containing the trained Gradient Boosting Regressor model.
* `requirements.txt`: List of required packages for the project.

# Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.
