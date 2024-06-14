import streamlit as st
import pandas as pd
import pickle

# Load the LabelEncoders
with open('label_house_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Load the trained model
with open('gbr_house_model.pkl', 'rb') as f:
    gb = pickle.load(f)

# Function to transform user input to match the model's input
def transform_input(user_input, label_encoders):
    user_input_encoded = user_input.copy()
    for col, encoder in label_encoders.items():
        if col in user_input.columns:
            user_input_encoded[col] = encoder.transform(user_input[[col]].values.ravel())
    return user_input_encoded

# Function to predict house price
def predict_price(user_input):
    user_input_transformed = transform_input(user_input, label_encoders)
    prediction = gb.predict(user_input_transformed.values.reshape(1, -1))[0]
    return prediction

# Streamlit app
st.write("""
<style>
    body {
        color: #333;
        font-family: Arial, sans-serif;
    }
    .container {
        max-width: 800px;
        margin: auto;
        padding: 20px;
    }
    h1 {
        color: #0078FF;
        text-align: center;
    }
    h2 {
        color: #333;
    }
    .prediction {
        background-color: #f4f4f4;
        padding: 10px;
        border-radius: 5px;
        margin-top: 20px;
    }
    .sidebar .sidebar-content h2 {
        color: white;
        font-weight: bold;
    }
</style>
<div class="container">
<h1>HOURICEP</h1>
<p>This app predicts the <strong>House Price</strong>!</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(
    '<h3 style="color:white;">Your House Price Prediction App</h3>', 
    unsafe_allow_html=True
)
# Sidebar for user input
st.sidebar.markdown(
    """
    <style>
        .sidebar .sidebar-content h2 {
            color: white;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown(
    '<h3 style="color:white;">Specify Input Parameters</h3>', 
    unsafe_allow_html=True
)

def user_input_features():
    POSTED_BY = st.sidebar.selectbox('POSTED_BY', ['Owner', 'Dealer', 'Builder'])
    UNDER_CONSTRUCTION = st.sidebar.selectbox('UNDER_CONSTRUCTION', [0, 1])
    RERA = st.sidebar.selectbox('RERA', [0, 1])
    BHK_NO = st.sidebar.slider('BHK_NO.', 1, 20, 2)
    BHK_OR_RK = st.sidebar.selectbox('BHK_OR_RK', ['BHK', 'RK'])
    SQUARE_FT = st.sidebar.number_input('SQUARE_FT', min_value=254, max_value=545500, value=500)
    READY_TO_MOVE = st.sidebar.selectbox('READY_TO_MOVE', [0, 1])
    RESALE = st.sidebar.selectbox('RESALE', [0, 1])
    LONGITUDE = st.sidebar.slider('LONGITUDE', -37.713008, 89.912884, 0.0)
    LATITUDE = st.sidebar.slider('LATITUDE', -121.761248, 152.962676, 0.0)
    CITY = st.sidebar.selectbox('CITY', ['Bangalore', 'Mysore', 'Ghaziabad', 'Kolkata', 'Kochi', 'Jaipur', 'Mohali', 'Chennai', 'Siliguri', 'Noida', 'Raigad', 'Bhubaneswar', 'Wardha', 'Pune', 'Mumbai', 'Nagpur', 'Deoghar', 'Bhiwadi', 'Faridabad', 'Lalitpur', 'Maharashtra', 'Vadodara', 'Visakhapatnam', 'Vapi', 'Mangalore', 'Aurangabad', 'Ottapalam', 'Vijayawada', 'Belgaum', 'Bhopal', 'Lucknow', 'Kanpur', 'Gandhinagar', 'Pondicherry', 'Agra', 'Ranchi', 'Gurgaon', 'Udupi', 'Indore', 'Jodhpur', 'Coimbatore', 'Valsad', 'Palghar', 'Surat', 'Varanasi', 'Guwahati', 'Amravati', 'Anand', 'Tirupati', 'Secunderabad', 'Raipur', 'Vizianagaram', 'Thrissur', 'Satna', 'Madurai', 'Chandigarh', 'Shimla', 'Gwalior', 'Rajkot', 'Sonipat', 'Allahabad', 'Berhampur', 'Roorkee', 'Dharuhera', 'Latur', 'Durgapur', 'Panchkula', 'Solapur', 'Durg', 'Goa', 'Jamshedpur', 'Hazaribagh', 'Jabalpur', 'Hosur', 'Morbi', 'Hubli', 'Karnal', 'Patna', 'Bilaspur', 'Ratnagiri', 'Meerut', 'Kotdwara', 'Jalandhar', 'Amritsar', 'Patiala', 'Ludhiana', 'Alwar', 'Kota', 'Panaji', 'Kolhapur', 'Ernakulam', 'Bhavnagar', 'Bharuch', 'Asansol', 'Jhansi', 'Margao', 'Anantapur', 'Eluru', 'Bhilai', 'Dehradun', 'Guntur', 'Jalgaon', 'Udaipur', 'Gurdaspur', 'Neemrana', 'Hassan', 'Sindhudurg', 'Hoshangabad', 'Kottayam', 'Dhanbad', 'Navsari', 'Bahadurgarh', 'Nellore', 'Dhule', 'Tirunelveli', 'Cuttack', 'Haridwar', 'Nainital', 'Jamnagar', 'Kanchipuram', 'Kadi', 'Karad', 'Jagdalpur', 'Panipat', 'Muzaffarpur', 'Salem', 'Jhunjhunu', 'Gandhidham', 'Junagadh', 'Moradabad', 'Ahmednagar', 'Jalna', 'Bhiwani', 'Palakkad', 'Kannur', 'Karjat', 'Akola', 'Jind', 'Gaya', 'Ambala', 'Ajmer', 'Hajipur', 'Dharwad', 'Pudukkottai', 'Kollam', 'Ooty', 'Bhandara', 'Barabanki', 'Rajpura', 'Palwal', 'Aligarh', 'Erode', 'Rudrapur', 'Tenali', 'Ongole', 'Nizamabad', 'Puri', 'Dalhousie', 'Siddipet', 'Solan', 'Darbhanga', 'Kadapa', 'Kakinada', 'Agartala', 'Warangal', 'Haldwani', 'Osmanabad', 'Bhagalpur', 'Bardhaman', 'Rishikesh', 'Chandrapur', 'Bokaro', 'Jharsuguda', 'Bhimavaram', 'Kurnool', 'Amroha', 'Hapur', 'Sabarkantha', 'Harda', 'Ujjain', 'Thoothukudi', 'Karaikudi', 'Mathura', 'Gadhinglaj', 'Rewari', 'Godhra', 'Kharagpur', 'Srikakulam', 'Srinagar', 'Midnapore', 'Rayagada', 'Banswara', 'Shirdi', 'Rohtak', 'Pali', 'Hathras', 'Yavatmal', 'Balasore', 'Chhindwara', 'Bareilly', 'Vidisha', 'Thanjavur', 'Kangra', 'Bikaner', 'Rewa', 'Porbandar', 'Nagaur', 'Nanded', 'Rourkela', 'Nadiad', 'Gulbarga', 'Palanpur', 'Bhadrak', 'Kurukshetra', 'Dibrugarh', 'Sagar', 'Machilipatnam', 'Pathanamthitta', 'Bankura', 'Jammu', 'Idukki', 'Korba', 'Raigarh', 'Silchar', 'Arrah', 'Nagaon', 'Karwar', 'Dahod', 'Nagapattinam', 'Sikar', 'Angul', 'Baddi', 'Darjeeling', 'Raisen', 'Hoshiarpur', 'Beed', 'Gadarwara', 'Jajpur', 'Haldia', 'Chittoor', 'Faizabad', 'Malappuram', 'Betul', 'Surendranagar', 'Phagwara', 'Visnagar', 'Rajnandgaon', 'Cuddalore', 'Raichur', 'Sambalpur', 'Gondia', 'Vellore', 'Bharatpur', 'Bhuj', 'Siwan', 'Washim'])
    
    data = {
        'POSTED_BY': POSTED_BY,
        'UNDER_CONSTRUCTION': UNDER_CONSTRUCTION,
        'RERA': RERA,
        'BHK_NO.': BHK_NO,
        'BHK_OR_RK': BHK_OR_RK,
        'SQUARE_FT': SQUARE_FT,
        'READY_TO_MOVE': READY_TO_MOVE,
        'RESALE': RESALE,
        'LONGITUDE': LONGITUDE,
        'LATITUDE': LATITUDE,
        'CITY': CITY
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_data = user_input_features()

# Display specified input parameters
st.markdown("""
<style>
    .parameters {
        border: 1px solid #ddd;
        padding: 20px;
        border-radius: 5px;
        background-color: #f9f9f9;
    }
    .sidebar .sidebar-content h2 {
        color: white;
        font-weight: bold;
    }
</style>
<div class="parameters">
<h2>Specified Input Parameters</h2>
</div>
""", unsafe_allow_html=True)
st.write(input_data)
st.write('---')

# Predict and display the house price
prediction = predict_price(input_data)
st.markdown(f"""
<style>
    .prediction {{
        background-color: #eaf4ff;
        padding: 20px;
        border-radius: 5px;
        text-align: center;
    }}
</style>
<div class="prediction">
<h2>Your house predicted price</h2>
</div>
""", unsafe_allow_html=True)
st.write(f"{prediction:.2f} Lakh")
st.write('---')
