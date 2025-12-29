import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
from sklearn.datasets import fetch_california_housing
st.title('🏠House Price prediction using ML🏡')

st.image('https://cdn.dribbble.com/userupload/23217976/file/original-203940eb89dec42ee25f6af662af24dc.gif')

df=pd.read_csv('house_data.csv')
X=df.iloc[:,:-3]
y=df.iloc[:,-1]

final_X = X
scaler = StandardScaler()
scaled_X = scaler.fit_transform(final_X)

st.sidebar.title('Select house features:')
st.sidebar.image('https://cdn.dribbble.com/userupload/25169331/file/original-fba6b47a5314c016c5c112e21d416bc2.gif')

all_value=[]
for i in final_X:
  result=st.sidebar.slider(f'Select{i} value')
  all_value.append(result)
st.write(all_value)
  








