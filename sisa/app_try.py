import streamlit as st
import pandas as pd
import numpy as np
import chart_studio.plotly as plotly
import plotly.figure_factory as ff
from plotly import graph_objs as go
from fbprophet import Prophet
from fbprophet.plot import plot_plotly




st.title('Data Kelahiran Jawa Barat')
st.write('Kelompok 1')
st.write('1.Alif Hafian fathurrahman'
	'2.Vincent wongso'
	'3.Nina Indah Gusmiarti')

dataset = ('lahir_hidup_laki_kab_bogor','lahir_hidup_perempuan_kab_bogor','lahir_mati_laki_kab_bogor','lahir_mati_perempuan_kab_bogor')	
option = st.selectbox('Select dataset for prediction',dataset)
DATA_URL =('./HISTORICAL_DATA/'+option+'.csv')

year = st.slider('Year of prediction:',1,10)
period = year * 365
#DATA_URL =('./HISTORICAL_DATA/3IINFOTECH_data.csv')

@st.cache
def load_data():
    data = pd.read_csv(DATA_URL)
    return data
	


data_load_state = st.text('Loading data...')
data = load_data()
data_load_state.text('Loading data... done!')

def plot_fig():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data.ds, y=data['y'], name="stock_open",line_color='deepskyblue'))
	fig.layout.update(title_text='Time Series data with Rangeslider',xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	return fig

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)
	
# plotting the figure of Actual Data
plot_fig()

# preparing the data for Facebook-Prophet.

data_pred = data[['ds','y']]


# code for facebook prophet prediction

m = Prophet()
m.fit(data_pred)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

#plot forecast
fig1 = plot_plotly(m, forecast)
if st.checkbox('Show forecast data'):
    st.subheader('forecast data')
    st.write(forecast)
st.write('Forecasting closing of stock value for'+option+' for a period of: '+str(year)+'year')
st.plotly_chart(fig1)

#plot component wise forecast
st.write("Component wise forecast")
fig2 = m.plot_components(forecast)
st.write(fig2)
	

	


