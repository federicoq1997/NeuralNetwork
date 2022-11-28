# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from datetime import date, timedelta
import statsmodels.tsa.api as smt
import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import pickle


# %%
def load_data():
	url = 'https://raw.githubusercontent.com/federicoq1997/NeuralNetwork/main/dataset_caffe.csv'
	df = pd.read_csv(url)
	return df


dataset = load_data()
dataset.head(10)

# %%
def monthly_sales(data):    
	data = data.copy()     
	# Drop the day indicator from the date column    
	data.date = data.date.apply(lambda x: str(x)[:-3])     
	# Sum sales per month    
	data = data.groupby('date')['qty'].sum().reset_index()
	data.date = pd.to_datetime(data.date)
	dates = [pd.to_datetime(i).date() for i in data['date'].unique()]
	range = pd.date_range('2019-10-01', data.tail(1)['date'].values[0], freq='MS')
	for i in range:
		if i.date() in dates:
			continue
		for (j,key) in enumerate(data.values):
			if hasattr(data, 'date') and data['date'][j].date() >  i.date():
				line = pd.DataFrame({"date": i, "qty": 0}, index=[j+1])
				data = pd.concat([data.iloc[:j], line, data.iloc[j:]]).reset_index(drop=True)
				break
	data.to_csv('./dataset/monthly_data.csv')     
	return data
monthly_data = monthly_sales(dataset)

# %%
#dataset.date = pd.to_datetime(dataset.date)
# monthly_data.date = pd.to_datetime(monthly_data.date)

# %%
def get_diff(data):
    data['qty_diff'] = data.qty.diff()    
    data = data.dropna()      
    return data
stationary_df = get_diff(monthly_data)
stationary_df.head(10)

# %%
def generate_arima_data(data):
    dt_data = data.set_index('date').drop('qty', axis=1)        
    dt_data.dropna(axis=0)     
    dt_data.to_csv('./dataset/arima_df.csv')
    return dt_data
arima_data = generate_arima_data(stationary_df)
arima_data.head(10)

# %%
def generate_supervised(data):
	supervised_df = data.copy()
	#create column for each lag
	for i in range(1,13):
			col = 'lag_' + str(i)
			supervised_df[col] = supervised_df['qty_diff'].shift(i)
	# print(supervised_df)	
	#drop null values
	supervised_df = supervised_df.dropna().reset_index(drop=True)
	supervised_df.to_csv('./dataset/model_df.csv', index=False)
	
	return supervised_df
model_df = generate_supervised(stationary_df)
model_df.head(10)

# %% [markdown]
# 
# ### Modelling<br>
#  <br>
# Train test split: separiamo i nostri dati in modo che gli ultimi 12 mesi facciano parte del set di test e il resto dei dati venga utilizzato per addestrare il nostro modello<br> 
# Scale the data: utilizzando uno scaler min-max, scaleremo i dati in modo che tutte le nostre variabili rientrano nell'intervallo da -1 a 1<br>
# Ridimensionamento inverso: dopo aver eseguito i nostri modelli, utilizzeremo questa funzione di supporto per invertire il ridimensionamento del passaggio 2<br>
# Crea un frame di dati di previsione: genera un frame di dati che include le vendite effettive acquisite nel nostro set di test e i risultati previsti dal nostro modello in modo da poter quantificare il nostro successo<br>
# Assegna un punteggio ai modelli: questa funzione di supporto salverà l'errore quadratico medio (RMSE) e l'errore assoluto medio (MAE) delle nostre previsioni per confrontare le prestazioni del nostro cinque modelli<br>
# 

# %%
from modeling.model import *




