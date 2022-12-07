# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 21:28:42 2022

@author: anony
"""

import numpy as np
import pickle
import streamlit as st
import pandas as pd
import datetime as dt
import joblib

loaded_model = pickle.load(open('E:/Egyetem/5. félév/GÉpi tan/feleves/regression.sav', 'rb'))

def prediction_function(input_data):


    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    #nem biztso, hogy kell
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    return prediction

def main():
    st.title("Microsoft Stock Prediction Webapp")
    
    df=pd.DataFrame(columns=['Date','Open', 'High', 'Low', 'Volume'])
    
    Open=st.number_input('Opening price')
    Open_float=float(Open)
    
    High=st.number_input('Highest price of the day')
    High_float=float(High)
    
    Low=st.number_input('Lowest price of the day')
    Low_float=float(Low)

    Volume=st.number_input('Volume')
    Volumeint=int(Volume)
    
    date=st.date_input('Day to predict')

    #most beadjuk az alap adatokat a dataframe-be

    df.loc['0'] = [date,Open_float,High_float, Low_float, Volumeint]
    
    #scaler beöltése (scaler_y), amivel inverz transzformáljuk a prediktált close értéket, mivel a modell skálázott adatra volt tanítva
    scaler_X =  pickle.load(open('E:/Egyetem/5. félév/GÉpi tan/feleves/scaler_X.sav', 'rb'))
    scaler_y = pickle.load(open('E:/Egyetem/5. félév/GÉpi tan/feleves/scaler_y.sav', 'rb'))
    
    inverse_close=''
    
    
    if st.button('Predict Closing Price'):
        #most az extra feature-öket adjuk be
        df['Date'] = pd.to_datetime(df['Date']).dt.date

        df['Date_str'] = df['Date'].astype(str)

        splitted_date=df['Date_str'].str.split('-', expand=True)

        df['day'] = splitted_date[2].astype('int')
        df['month'] = splitted_date[1].astype('int')
        df['year'] = splitted_date[0].astype('int')

        #dátum átkonvertálása
        df['Date'] = pd.to_datetime(df['Date'])
        df['Date']=df['Date'].map(dt.datetime.toordinal)
        
        #segéd kidobása
        df=df.drop(columns=['Date_str'])
        
        #még egy feautre hozzáadása
        df['is_quarter_end'] = np.where(df['month']%3==0,1,0)
        
        #eddig megvan a full dataframe az extra feature-ökkel
        st.dataframe(df)
        
        #skálázom a dataframe-et
        scaled_df=pd.DataFrame(scaler_X.transform(df))
        
        
        #prediktálás
        scaled_close=prediction_function(scaled_df)
        inverse_close=scaler_y.inverse_transform(scaled_close)
        
        
    st.success(inverse_close)
    
if __name__ == '__main__':
    main()