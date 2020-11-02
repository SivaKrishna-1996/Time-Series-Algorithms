# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 14:36:23 2020

@author: Manoj
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 15 12:50:04 2020
@author: krish.naik
"""

from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score
from fbprophet import Prophet

app=Flask(__name__)

@app.route('/')
def welcome():
    return "Welcome All"


@app.route('/predict',methods=["Get"])
def predict_note_file():
    
    # df_test=pd.read_csv(request.files.get("file"))
    dt=pd.read_csv('Final Data1.csv')
    #print(df_test.head())
    s1=request.args.get('s1')
    s2=request.args.get('s2')
    s3=request.args.get('s3')
    s4=request.args.get('s4')
    s5=request.args.get('s5')
    s6=request.args.get('s6')
    s7=request.args.get('s7')
    t=int(request.args.get('s8'))
    dl1=dt[(dt[s6]==s7) & (dt[s4]==s5)] 
    datewise=dl1
    # datewise= datewise.astype('float32')
    model_train=datewise.iloc[:int(datewise.shape[0]*0.95)]
    valid=datewise.iloc[int(datewise.shape[0]*0.95):]
    n11=pd.infer_freq(datewise[s1], warn=True)
    
    #Prophet
    datewise1=datewise.reset_index()
    datewise1.rename(columns={s1: 'ds',s3: 'y'},inplace=True)
    train=datewise1.iloc[:int(datewise1.shape[0]*0.95)]
    valid=datewise1.iloc[int(datewise1.shape[0]*0.95):]
    m=Prophet(weekly_seasonality=True)
    m.fit(train)
    future=m.make_future_dataframe(periods=len(valid),freq=n11)
    forecast=m.predict(future)
    predictions=forecast.tail(len(valid))['yhat']
    # print('\n')
    # print("Root Mean Squared Error for Prophet Model: ",rmse(valid['y'],predictions))
    # print('\n')
    # list9.append(rmse(valid['y'],predictions))
    m=Prophet(weekly_seasonality=True)
    m.fit(datewise1)
    future=m.make_future_dataframe(periods=t,freq=n11)
    forecast=m.predict(future)
    forecast_prophet=forecast['yhat'].tail(t)
    forecast_prophet1=forecast['yhat_lower'].tail(t)
    forecast_prophet2=forecast['yhat_upper'].tail(t)
    
    
    
    return str([list(forecast_prophet),list(forecast_prophet1),list(forecast_prophet2)])

if __name__=='__main__':
    app.run(host='0.0.0.0',port=8000)
