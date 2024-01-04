import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.metrics import dtw, dtw_path
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import silhouette_score
import random
import tqdm
import re

from statsforecast import StatsForecast
from statsforecast.models import CrostonClassic
from statsforecast import StatsForecast
from statsforecast.models import ADIDA, CrostonClassic, IMAPA, TSB


def replace_amount(x):
    if '만원 이하' in x:
        min_val = int(re.findall(r'\d+', x)[0])
        max_val = int(re.findall(r'\d+', x)[1])

        return (min_val + max_val) / 2
    else:
        return 1000

def preprocess(buy_data):

    buy_data = buy_data.dropna()

    buy_data["금액(중간값)"] = buy_data['금액'].apply(replace_amount)
    buy_data['매출일자(배송완료일자)'] = pd.to_datetime(buy_data['매출일자(배송완료일자)'], format='%Y%m%d')
    buy_data["년도"] = buy_data['매출일자(배송완료일자)'].apply(lambda x : x.year)
    buy_data["년월"] = buy_data['매출일자(배송완료일자)'].apply(lambda x : x.strftime("%Y-%m"))

    return buy_data

def make_timeseries(data):
    time_series = pd.pivot_table(data,index=["고객ID"], values = ["금액(중간값)"],columns = ["년도"],aggfunc=["sum"])
    time_series = time_series.T.reset_index().drop(["level_0","level_1"],axis=1).T
    time_series.columns = time_series.loc["년도"]
    time_series.drop(["년도"],axis=0,inplace=True)
    time_series = time_series.fillna(0)

    return time_series

def make_predict(time_series):
    sim_df = time_series.reset_index().melt(id_vars=["고객ID"],value_name='금액(중간값)').sort_values(by=["고객ID","년도"])

    sim_df = sim_df[["고객ID","년도","금액(중간값)"]]
    sim_df.columns = ["unique_id","ds","y"]
    train = sim_df.loc[sim_df["ds"]<2023]
    valid = sim_df.loc[sim_df["ds"]==2023]

    model = StatsForecast(models=[ ADIDA(), 
                                CrostonClassic(), 
                                IMAPA(), 
                                TSB(alpha_d=0.2, alpha_p=0.2)], freq='Y', n_jobs=-1)

    model.fit(train)

    p = model.predict(h=1)
    p["ds"] = [2023]*len(p)

    return p

def buy_term_mean(data):
    df = data.copy()
    df.sort_values(['고객ID', '매출일자(배송완료일자)'], inplace=True)
    df['구매일자_이전'] = df.groupby('고객ID')['매출일자(배송완료일자)'].shift(1)
    df['구매간격'] = (df['매출일자(배송완료일자)'] - df['구매일자_이전']).dt.days
 
    평균_구매_간격 = df.groupby('고객ID')['구매간격'].mean()

    mean_term = pd.DataFrame(평균_구매_간격)
    
    return mean_term

def buy_term_timeseries(data):
    df = data.copy()
    df.sort_values(['고객ID', '매출일자(배송완료일자)'], inplace=True)
    df['구매일자_이전'] = df.groupby('고객ID')['매출일자(배송완료일자)'].shift(1)
    df['구매간격'] = (df['매출일자(배송완료일자)'] - df['구매일자_이전']).dt.days
    df['구매순서'] = df.groupby('고객ID').cumcount() + 1
    df = df.reset_index(drop=True)
    
    평균_구매_간격 = df.groupby('고객ID')['구매간격'].mean()
    df_ts = pd.DataFrame(평균_구매_간격)
    
    for order in range(2, 9):
        df_order = df[df['구매순서'] == order].reset_index()

        for index, row in df_order.iterrows():
            customer_id = row['고객ID']
            purchase_interval = df_order.loc[df_order['고객ID'] == customer_id, '구매간격'].values[0]
            term_col = f'term{order - 1}'
            df_ts.at[index, term_col] = purchase_interval
            
    df_ts = df_ts.fillna(0)
    
    return df_ts
