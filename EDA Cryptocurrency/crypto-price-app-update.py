# This app is for educational purpose only. Insights gained is not financial advice. Use at your own risk!
import streamlit as st
from PIL import Image
import pandas as pd
import base64
import matplotlib.pyplot as plt
import time
import httpx
import json
#---------------------------------#
# New feature (make sure to upgrade your streamlit library)
# pip install --upgrade streamlit

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(layout="wide")
#---------------------------------#
# Title

image = Image.open('logo.jpg')

st.image(image, width = 600)

st.title('Crypto Price App')
st.markdown("""
This app retrieves cryptocurrency prices for the top 100 cryptocurrency from the **livecoinwatch**!

""")
#---------------------------------#
# About
expander_bar = st.expander("About")
expander_bar.markdown("""
* **Python libraries:** base64, pandas, streamlit, Image, httpx, matplotlib, json, time
* **Data source:** [livecoinwatch](https://www.livecoinwatch.com/).
""")

#---------------------------------#
# Page layout (continued)
## Divide page to 3 columns (col1 = sidebar, col2 and col3 = page contents)
col1 = st.sidebar
col2, col3 = st.columns((2,1))

#---------------------------------#
# Sidebar + Main panel
col1.header('Input Options')

## Sidebar - Currency price unit
currency_price_unit = col1.selectbox('Select currency for price', ('USD', 'ETH', 'BNB'))

# Web scraping of CoinMarketCap data
# Fungsi untuk expand kolom yang berisi dictionary menggunakan JSON
def expand_dict_col(col, prefix):
  return col.apply(lambda x: pd.Series(x)).add_prefix(prefix)

@st.cache_data
def load_data(currency=currency_price_unit):
  base_url = "https://http-api.livecoinwatch.com/coins"
  params = {
      "offset" : 0,
      "limit" : 100,
      "sort" : "rank",
      "order" : "ascending",
      "currency" : currency
  }

  response = httpx.get(base_url, params=params)
  if response.status_code != 200:
    print(response.raise_for_status())
    return

  #response.json().keys()
  df = pd.DataFrame(response.json()["data"])
  #df.to_csv("crypto table.csv", index=False)

  # Daftar kolom yang ingin dihapus, termasuk satu kolom yang tidak ada di DataFrame
  columns_to_drop = ['elisted', 'plot', 'minibook', 'color', 'votes', 'trending', 
                     'pairs', 'ico', 'holders', 'transfers', 'reddit', 'book'] 
  
  # Menghapus beberapa kolom sekaligus tanpa loop, mengabaikan kolom yang tidak ada
  df = df.drop(columns=columns_to_drop, errors='ignore')
  
  # Daftar kolom yang berisi dictionary dan prefiks yang akan digunakan
  columns_to_expand = {
      'delta': 'delta_',
      'deltav': 'deltav_',
      'extremes': 'extremes_'
      }
  
  # Loop untuk mengembangkan kolom dictionary
  expanded_dfs = []
  for col, prefix in columns_to_expand.items():
    expanded_dfs.append(expand_dict_col(df[col], prefix))
  
  # Menggabungkan kembali kolom ID dan kolom yang sudah diekspansi
  df = pd.concat([df.drop(columns=columns_to_expand.keys())] + expanded_dfs, axis=1)
  
  # Daftar kolom yang perlu dioperasikan
  columns_to_update = [
      'delta_hour', 'delta_day', 'delta_week', 'delta_month', 'delta_quarter', 'delta_year',
      'deltav_hour', 'deltav_day', 'deltav_week', 'deltav_month', 'deltav_quarter', 'deltav_year'
      ]
  
  # Loop untuk menerapkan operasi pada setiap kolom dalam daftar
  for col in columns_to_update:
    new_col = col + '_%'
    df[new_col] = (df[col] - 1) * 100
    df.drop(columns=[col], inplace=True)
  return df

df = load_data(currency=currency_price_unit)

## Sidebar - Cryptocurrency selections
sorted_coin = sorted( df['code'] )
selected_coin = col1.multiselect('Cryptocurrency', sorted_coin, sorted_coin)

df_selected_coin = df[ (df['code'].isin(selected_coin)) ] # Filtering data

## Sidebar - Number of coins to display
num_coin = col1.slider('Display Top N Coins', 1, 100, 100)
df_coins = df_selected_coin[:num_coin]

## Sidebar - Percent change timeframe
percent_timeframe = col1.selectbox('Percent change time frame',
                                    ['1y', '3m', '1m', '7d', '24h', '1h'])
percent_dict = {"1y":'delta_year_%', "3m":'delta_quarter_%', "1m":'delta_month_%', 
                "7d":'delta_week_%', "24h":'delta_day_%', "1h":'delta_hour_%'}
selected_percent_timeframe = percent_dict[percent_timeframe]

## Sidebar - Sorting values
sort_values = col1.selectbox('Sort values?', ['Yes', 'No'])

col2.subheader('Price Data of Selected Cryptocurrency')
col2.write('Data Dimension: ' + str(df_coins.shape[0]) + ' rows and ' + str(df_coins.shape[1]) + ' columns')

col2.dataframe(df_coins)

# Download CSV data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="crypto.csv">Download CSV File</a>'
    return href

col2.markdown(filedownload(df_selected_coin), unsafe_allow_html=True)

#---------------------------------#
# Preparing data for Bar plot of % Price change
col2.subheader('Table of % Price Change')
df_change = pd.concat([df_coins['code'], df_coins['delta_hour_%'], df_coins['delta_day_%'], 
                        df_coins['delta_week_%'], df_coins['delta_month_%'], df_coins['delta_quarter_%'], 
                        df_coins['delta_year_%']], axis=1)
df_change = df_change.set_index('code')

#Daftar kolom yang akan diperiksa
columns_to_check = [
    'delta_hour_%', 'delta_day_%', 'delta_week_%', 'delta_month_%', 'delta_quarter_%', 'delta_year_%'
]

# Loop untuk menambahkan kolom 'positive_delta_*'
for col in columns_to_check:
    new_col = 'positive_' + col
    df_change[new_col] = df_change[col] > 0

col2.dataframe(df_change)

# Conditional creation of Bar plot (time frame)
col3.subheader('Bar plot of % Price Change')

if percent_timeframe == '1y':
    if sort_values == 'Yes':
        df_change = df_change.sort_values(by=['delta_year_%'])
    col3.write('*1 year period*')
    plt.figure(figsize=(5,25))
    plt.subplots_adjust(top = 1, bottom = 0)
    df_change['delta_year_%'].plot(kind='barh', color=df_change['positive_delta_year_%'].map({True: 'g', False: 'r'}))
    col3.pyplot(plt)
elif percent_timeframe == '3m':
    if sort_values == 'Yes':
        df_change = df_change.sort_values(by=['delta_quarter_%'])
    col3.write('*3 months period*')
    plt.figure(figsize=(5,25))
    plt.subplots_adjust(top = 1, bottom = 0)
    df_change['delta_quarter_%'].plot(kind='barh', color=df_change['positive_delta_quarter_%'].map({True: 'g', False: 'r'}))
    col3.pyplot(plt)
elif percent_timeframe == '1m':
    if sort_values == 'Yes':
        df_change = df_change.sort_values(by=['delta_month_%'])
    col3.write('*1 month period*')
    plt.figure(figsize=(5,25))
    plt.subplots_adjust(top = 1, bottom = 0)
    df_change['delta_month_%'].plot(kind='barh', color=df_change['positive_delta_month_%'].map({True: 'g', False: 'r'}))
    col3.pyplot(plt)
elif percent_timeframe == '7d':
    if sort_values == 'Yes':
        df_change = df_change.sort_values(by=['delta_week_%'])
    col3.write('*7 days period*')
    plt.figure(figsize=(5,25))
    plt.subplots_adjust(top = 1, bottom = 0)
    df_change['delta_week_%'].plot(kind='barh', color=df_change['positive_delta_week_%'].map({True: 'g', False: 'r'}))
    col3.pyplot(plt)
elif percent_timeframe == '24h':
    if sort_values == 'Yes':
        df_change = df_change.sort_values(by=['delta_day_%'])
    col3.write('*24 hour period*')
    plt.figure(figsize=(5,25))
    plt.subplots_adjust(top = 1, bottom = 0)
    df_change['delta_day_%'].plot(kind='barh', color=df_change['positive_delta_day_%'].map({True: 'g', False: 'r'}))
    col3.pyplot(plt)
else:
    if sort_values == 'Yes':
        df_change = df_change.sort_values(by=['delta_hour_%'])
    col3.write('*1 hour period*')
    plt.figure(figsize=(5,25))
    plt.subplots_adjust(top = 1, bottom = 0)
    df_change['delta_hour_%'].plot(kind='barh', color=df_change['positive_delta_hour_%'].map({True: 'g', False: 'r'}))
    col3.pyplot(plt)
