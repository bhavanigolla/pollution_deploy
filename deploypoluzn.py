import streamlit as st
import pandas as pd
import pickle 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
data_path='pollution_dataset.csv'
pollution=pd.read_csv(data_path)
pollution.rename(columns={'PM2.5':'PM25','Air Quality':'Air_Quality'},inplace=True)
model=RandomForestClassifier()
x=pollution.drop('Air_Quality',axis=1)
y=pollution['Air_Quality']
model.fit(x,y)
model_path='pollution.pkl'
with open(model_path,'wb') as f:
  pickle.dump(model,f)
  
# pollution.rename(columns={'PM2.5':'PM25','Air Quality':'Air_Quality'},inplace=True)
st.title('POLLUTION PREDICTION')
st.subheader('SLIDE THE BELOW VALUES TO PREDICT POLLUTION')
Temperature=st.slider('Temperature value:',min_value=1.0,max_value=200.0,step=1.0)
Humidity=st.slider('Humidity value:',min_value=1.0,max_value=200.0,step=1.0)
PM25=st.slider('PM25 value:',min_value=1.0,max_value=200.0,step=1.0)
PM10=st.slider('PM10 value:',min_value=1.0,max_value=200.0,step=1.0)
NO2=st.slider('NO2 value:',min_value=1.0,max_value=200.0,step=1.0)
SO2=st.slider('SO2 value:',min_value=1.0,max_value=200.0,step=1.0)
CO=st.slider('CO value:',min_value=1.0,max_value=200.0,step=1.0)
Proximity_to_Industrial_Area=st.slider('Proximity_to_Industrial_Area value:',min_value=1.0,max_value=200.0,step=1.0)
Population_Density=st.slider('Population_Density value:',min_value=1,max_value=500,step=1)
input=np.array([[ Temperature, Humidity , PM25, PM10, NO2, SO2, CO,Proximity_to_Industrial_Area, Population_Density]])
with open(model_path,'rb') as f:
  model1=pickle.load(f)
st.checkbox('im not the robot')
if st.button('Predict'):
  prediction=model1.predict(input)[0]
  st.write(f"Predicted Air Quality: {prediction}")

#input=np.array[[Temperature,Humidity,PM2.5,PM10,NO2,SO2,CO,Proximity_to_Industrial_Areas,Population_Density,Air Quality]][0]