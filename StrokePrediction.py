import pandas as pd 
import numpy as np 
import seaborn as sns 
import streamlit as st 
import matplotlib.pyplot as plt 
import joblib
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv('healthcare-dataset-stroke-data.csv')
df = data.copy()

scaler = StandardScaler()
encoder = LabelEncoder()

# # copy your data
# new_data = data.copy()
# num= new_data.select_dtypes(include = 'number')
# cat= new_data.select_dtypes(exclude = 'number')

encoded = {}
# Encode the categorical data set
for i in df.select_dtypes(exclude = 'number').columns:
    encoder = LabelEncoder()
    df[i] = encoder.fit_transform(df[i])
    encoded[i] = encoder


st.markdown("<h2 style = 'color: #561C24; text-align: center; font-family: helvetica '>Stroke Prediction</h2>", unsafe_allow_html =True)
st.markdown('<br>', unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #6D2932; text-align: center; font-family: helvetica'>Designed and built by", unsafe_allow_html =True)
st.markdown('<br>', unsafe_allow_html = True)
st.markdown("<h3 style = 'margin: -30px; color: #1E1E1E; text-align: center; font-family: cursive '>kaka Tech Word", unsafe_allow_html =True)

st.markdown("<h4 style = 'margin: -30px; color: #1E1E1E; text-align: center; font-family: cursive '>Giving solution to mankind", unsafe_allow_html =True)
st.markdown('<br>', unsafe_allow_html = True)
st.image('72zF.gif', width = 100, use_column_width=True)

st.sidebar.markdown("""
    <div style="display: flex; justify-content: center;">
<h1>DATA</h1>
    </div>
""", unsafe_allow_html=True)
st.sidebar.image('heartcheck.png')
st.sidebar.markdown("<h1 style = 'color: #561C24; text-align: center; font-family: cursive '>Project Overview</h1>", unsafe_allow_html =True)
st.sidebar.markdown("<h6 style = 'color: #1E1E1E; text-align: center; font-family: 'Arial', sans-serif '>This model aims to predict the likelihood of stroke based on several features, including heart problems, age, gender, occupation, smoking habits, and other relevant health issues. By analyzing a dataset with these parameters, the model employs machine learning algorithms to identify patterns and correlations that contribute to the prediction of stroke risk. The predictive nature of the model is valuable for early detection and prevention, enabling healthcare professionals to take proactive measures for individuals at higher risk of stroke. It provides a tool for personalized health assessments and interventions, emphasizing the importance of targeted preventive measures in mitigating stroke risks.</h6>", unsafe_allow_html =True)
st.sidebar.markdown("<h1 style = 'color: #561C24; text-align: center; font-family: cursive '>Dataframe</h1>", unsafe_allow_html =True)
st.sidebar.dataframe(data,use_container_width=True )
st.sidebar.markdown('<br>', unsafe_allow_html = True)

col1, col2, col3 = st.columns(3)


# Add content to each column
with col1:
    st.markdown("<h4 style='color: #561C24;'>GENDER</h3>",unsafe_allow_html=True)
    gender = st.selectbox("Whats your gender", data['gender'].unique())
   

with col2:
    st.markdown("<h4 style='color: #561C24;'>AGE</h3>",unsafe_allow_html=True)
    age = st.number_input("Whats your age")
with col3:
    st.markdown("<h4 style='color: #561C24;'>HYPERTENSION</h3>", unsafe_allow_html=True)
    hypertension = st.selectbox('Are you hypertensive:1=yes,0=no',data['hypertension'].unique())

col4, col5, col6 = st.columns(3)
with col4:
    st.markdown("<h5 style='color: #561C24;'>HEART DISEASE</h3>",unsafe_allow_html=True)
    heart_disease = st.selectbox("DO you have a heart disease:1=yes,0=no", data['heart_disease'].unique())
   

with col5:
    st.markdown("<h5 style='color: #561C24;'>MARITAL STATUS</h3>",unsafe_allow_html=True)
    ever_married = st.selectbox("Have you ever_married",data['ever_married'].unique())
with col6:
    st.markdown("<h5 style='color: #561C24;'>OCCUPATION CARTEGORY</h3>", unsafe_allow_html=True)
    work_type = st.selectbox('whats your work type',data['work_type'].unique())
col7, col8, col9= st.columns(3)
with col7:
    st.markdown("<h5 style='color: #561C24;'>TYPE OF RESIDENCE</h3>",unsafe_allow_html=True)
    Residence_type = st.selectbox("Where do you reside", data['Residence_type'].unique())
   

with col8:
    st.markdown("<h5 style='color: #561C24;'>AVERAGE GLUCOSE LEVEL</h3>",unsafe_allow_html=True)
    avg_glucose_level = st.number_input("What is the avg. glucose level")
with col9:
    st.markdown("<h5 style='color: #561C24;'>BODY MASS INDEX</h3>",unsafe_allow_html=True)
    bmi = st.number_input("what is your BMI")
st.markdown("<h5 style='color: #561C24;'>SMOKING STATUS</h3>",unsafe_allow_html=True)
smoking_status = st.selectbox("Do you smoke",data['smoking_status'].unique() )

#TENURE = st.selectbox("duration in the network", data['TENURE'].unique())



model = joblib.load('stroke.pkl')
input_var = pd.DataFrame({'gender':[gender], 'age':[age], 'hypertension':[hypertension], 'heart_disease':[heart_disease], 'ever_married':[ever_married],'work_type':[work_type], 'Residence_type':[Residence_type], 'avg_glucose_level':[avg_glucose_level], 'bmi':[bmi],'smoking_status':[smoking_status]})


input_var['gender'] = encoded['gender'].transform(input_var['gender'])
input_var['ever_married'] = encoded['ever_married'].transform(input_var['ever_married'])
input_var['work_type'] = encoded['work_type'].transform(input_var['work_type'])
input_var['Residence_type'] = encoded['Residence_type'].transform(input_var['Residence_type'])
input_var['smoking_status'] = encoded['smoking_status'].transform(input_var['smoking_status'])
st.dataframe(input_var)



predicted = model.predict(input_var)
output = None
if predicted[0] == 0:
    output = 'Not have Stroke'
else:
    output = 'have Stroke'

prediction, interprete = st.tabs(["Model Prediction", "Model Interpretation"])
with prediction:
    pred = st.button('Push To Predict')
    if pred: 
        st.success(f'The patient is predicted to {output}')
        st.balloons()
# predicter = st.button('Predict')
# if predicter:
#     prediction = model.predict(input_var)
#     output = None
# if prediction[0] == 0:
#     output = 'Not Churn'
# else:
#     output = 'Churn'

#     st.success(f'The Predicted value for your company is {prediction}')
#     st.balloons()
