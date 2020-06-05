import sys, os

import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from talib import BBANDS
from talib import MACD

from urllib.request import urlretrieve
from urllib.parse import quote
from unittest.mock import inplace

from cv2 import cv2
from plotly.graph_objs import candlestick


ALPHAVANTAGE_API = "JFG65ECO4YHPY5HM"

def retrieve_data(func, symbol, inter, outputsize="full", datatype="csv"):
    if not os.path.exists("StockData"):
        os.mkdir("StockData")
    func = quote(func)
    symbol = quote(symbol)
    if inter is not None:
        inter = quote(inter)
    outputsize = quote(outputsize)
    datatype = quote(datatype)
    
    if inter is not None:
        alphavantage_url = "https://www.alphavantage.co/query?function={func}&symbol={symbol}&interval={inter}&outputsize={outputsize}&apikey={apikey}&datatype={datatype}".format(func=func, symbol=symbol, inter=inter, outputsize=outputsize, apikey=ALPHAVANTAGE_API, datatype=datatype)
    else:
        alphavantage_url = "https://www.alphavantage.co/query?function={func}&symbol={symbol}&outputsize={outputsize}&apikey={apikey}&datatype={datatype}".format(func=func, symbol=symbol, inter=inter, outputsize=outputsize, apikey=ALPHAVANTAGE_API, datatype=datatype)
    # alphavantage_url = alphavantage_url.encode("utf-8")
    csv_name = "StockData/{symbol}.csv".format(symbol=symbol, inter=func)
    
    urlretrieve(alphavantage_url, csv_name)
    df = pd.read_csv(csv_name)
    
    return df

def retrieve_etf(funds):
    for etf in funds:
        df = retrieve_data("TIME_SERIES_DAILY_ADJUSTED", etf, None, "full")
        print("{fund} history downloaded.".format(fund=etf))
        
def get_indicators(df, indicators=[]):
    if "bbands" in indicators:
        df["upperband"], df["middleband"], df["lowerband"] = BBANDS(df["close"], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    if "macd" in indicators:
        df["macd"], df["macdsignal"], df["macdhist"] = MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)
    if "price_logreturn" in indicators:
        df["price_logreturn"] = log_return(df["close"].to_numpy())
    if "volume_logreturn" in indicators:
        df["volume_logreturn"] = log_return(df["volume"].to_numpy())
        
    return df

def check_indicators(df, indicators):
    required = []
    
    if "bbands" in indicators:
        required.append("upperband")
        required.append("lowerband")
        required.append("middleband")
    if "macd" in indicators:
        required.append("macd")
        required.append("macdsignal")
        required.append("macdhist")
    if "price_logreturn" in indicators:
        required.append("price_logreturn")
    if "volume_logreturn" in indicators:
        required.append("volume_logreturn")    
    
    return all(attribute in df.columns for attribute in required)

def log_return(ts):
    assert len(ts) > 2, "List must contian at least 2 values."
    return np.insert(np.array([np.log10(ts[i]/ts[i-1]) for i in range(1, len(ts))]), 0, 0, axis=0)

def generate_timeserie(df, window, start_idx, end_idx, indicators=[]):
    assert start_idx >= window, "Invalid range of index. Make sure start index is greater than window size."
    end_idx = len(df) if end_idx > len(df) else end_idx
    
    if not pd.Series(['price_logreturn', 'volume_logreturn']).isin(df.columns).all():
        df = get_indicators(df, indicators)
    timeseries = []
    for i in range(start_idx, end_idx):
        df_window = df[i-window:i]
        timeseries.append(np.stack((df_window["price_logreturn"], df_window["volume_logreturn"]), axis=1))
        # print("Time series {start}-{end} generated.".format(start=df.loc[i-window, 'timestamp'], end=df.loc[i-1, 'timestamp']))
#    print(np.array(timeseries).shape)
    return timeseries   
        
def generate_chart(df, symbol, window, start_idx, end_idx, indicators=[], overwrite=False):
    assert start_idx >= window, "Invalid range of index. Make sure start index is greater than window size."
    end_idx = len(df) if end_idx > len(df) else end_idx
    df.reset_index(inplace=True, drop=True)
    
    if not pd.Series(indicators).isin(df.columns).all():
        df = get_indicators(df, indicators)
    if not os.path.isdir(symbol):
        os.mkdir(symbol)
    charts = []
    for i in range(start_idx, end_idx):
        if not overwrite:
            time_window = "{start}_to_{end}".format(start=df.loc[i-window, 'timestamp'], end=df.loc[i-1, 'timestamp'])
            chart_name = "{folder}/{symbol}_{index}_MERGED.png".format(folder=symbol, symbol=symbol, index=time_window)
            if os.path.exists(chart_name):
                # print("{name} already exist.".format(name=chart_name))
                chart = cv2.imread(chart_name)
                charts.append(chart)
                continue
        
        df_window = df[i-window:i]
        
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0, row_heights=[4, 1, 1])
        
        fig_candlestick = go.Candlestick(x=list(range(window)),
                                         open=df_window["open"],
                                         high=df_window["high"],
                                         low=df_window["low"],
                                         close=df_window["close"],
                                         increasing_line_color="blue", decreasing_line_color="red",
                                         opacity=1)
        fig.append_trace(fig_candlestick, row=1, col=1)
        
        if "bbands" in indicators:
            fig.add_trace(go.Scatter(y=df_window["upperband"], mode="lines", line_color="grey"), row=1, col=1)
            fig.add_trace(go.Scatter(y=df_window["middleband"], fill="tonexty", mode="lines", line_color="grey"), row=1, col=1)
            fig.add_trace(go.Scatter(y=df_window["lowerband"], fill="tonexty", mode="lines", line_color="grey"), row=1, col=1)
             
        if "macd" in indicators:
            fig.add_trace(go.Scatter(y=df_window["macd"], mode="lines", line_color="magenta"), row=2, col=1)
            fig.add_trace(go.Scatter(y=df_window["macdsignal"], mode="lines", line_color="yellow"), row=2, col=1)
            pos_hist = df_window.copy()
            pos_hist[ pos_hist["macdhist"] > 0] = 0
            fig.add_trace(go.Bar(x=list(range(window)), y=pos_hist["macdhist"], marker_color="orange"), row=2, col=1)
            neg_hist = df_window.copy()
            neg_hist[ neg_hist["macdhist"] < 0] = 0
            fig.add_trace(go.Bar(x=list(range(window)), y=neg_hist["macdhist"], marker_color="cyan"), row=2, col=1)
              
        fig.add_trace(go.Bar(x=list(range(window)), y=df["volume"], marker_color="green"), row=3, col=1)
        
        fig.update_layout(
            xaxis=dict(
                autorange=True,
                showgrid=False,
                ticks='',
                showticklabels=False
            ),
            yaxis=dict(
                autorange=True,
                showgrid=False,
                ticks='',
                showticklabels=False
            ),
            yaxis2=dict(
                autorange=True,
                showgrid=False,
                ticks='',
                showticklabels=False
            ),
            yaxis3=dict(
                autorange=True,
                showgrid=False,
                ticks='',
                showticklabels=False
            ),
            showlegend=False,
             plot_bgcolor='rgb(255, 255, 255)',
            xaxis_rangeslider_visible=False
        )
        
        fig.write_image(chart_name)
        
        merged = cv2.imread(chart_name)
        merged = cv2.resize(merged, (112, 112))
        cv2.imwrite(chart_name, merged)
        # print("{name} created.".format(name=chart_name))
        charts.append(merged)
        
#         if i-window == 5:
#             break
    
    return charts
    
if __name__ == '__main__':          
    selected_etf = ["SPY", "QQQ", "XLU", "XLE", "XLP", "XLY", "XLF", "EWZ", "EWH"]
    # retrieve_etf(selected_etf)
            
#     df = pd.read_csv("StockData/SPY.csv")
#     df = df.iloc[::-1]
#     df.reset_index(inplace=True)
#     for symbol in selected_etf:
#         generate_chart(df, symbol, 30, 30, len(df), indicators=["bbands", "macd"])
    # generate_timeserie(df, 30, 30, len(df), indicators=["price_logreturn", "volume_logreturn"])
    
    
    # spy_df = retrieve_data("TIME_SERIES_DAILY", "SPY", None, "full")
    # qqq_df = retrieve_data("TIME_SERIES_DAILY", "QQQ", None, "full")
    # xlu_df = retrieve_data("TIME_SERIES_DAILY", "XLU", None, "full")
    # xle_df = retrieve_data("TIME_SERIES_DAILY", "XLE", None, "full")
    # xlp_df = retrieve_data("TIME_SERIES_DAILY", "XLP", None, "full")
    # xly_df = retrieve_data("TIME_SERIES_DAILY", "XLY", None, "full")
    # xlf_df = retrieve_data("TIME_SERIES_DAILY", "XLF", None, "full")
    # ewz_df = retrieve_data("TIME_SERIES_DAILY", "EWZ", None, "full")
    # ewh_df = retrieve_data("TIME_SERIES_DAILY", "EWH", None, "full")


 