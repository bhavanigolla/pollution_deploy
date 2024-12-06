import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
data_path='pollution_dataset.csv'
pollution=pd.read_csv(data_path)
# print(pollution.head())
# print(pollution.columns)
X = pollution.drop('Air Quality', axis=1)
y = pollution['Air Quality']
model = RandomForestClassifier()
model.fit(X, y)
model_path='pollution.pkl'
with open(model_path,'wb') as f:
  pickle.dump(model,f)
sample_data=[[223.9	, 51.9	, 14.7 ,	24.3 ,	5.2	, 12.6 , 1.24,	4.5	, 282	]]
predictions= model.predict(sample_data)[0]
print("Prediction:",predictions)
