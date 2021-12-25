##!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import pymysql
from sqlalchemy import create_engine
pymysql.install_as_MySQLdb()
import time
import datetime
from numpy import inf
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance,plot_tree
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


###########数据处理##########
#缺失率删除
def info(df):
    stats = []
    for col in df.columns:
        stats.append((col, df[col].nunique(), df[col].isnull().sum() * 100 / df.shape[0], df[col].value_counts(normalize=True, dropna=False).values[0] * 100, df[col].dtype))
    stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values', 'Per_biggest_category', 'type'])
    tt=stats_df.sort_values('Percentage of missing values', ascending=False)[:]
    return tt

def drop_na_unique(df,miss_rate):
    '''
    目标：缺失值清洗
    df：目标数据集
    miss_rate：缺失率阈值
    '''
    na = info(df)
    na_fea = list(na[na['Percentage of missing values']>miss_rate]['Feature'])
    df = df.drop(na_fea,axis=1)
    return df

def coding(df,fea,code_type):
    '''
    编码
    df：数据集
    fea：特征
    code_type：编码类型
    '''
    if code_type =='LabelEncoding':
        #labelencoder
        df[fea] = pd.factorize(df[fea])[0]
        return df
    elif code_type =='CountEncoding': 
        #频数编码
        df[fea] = df[fea].map(df[fea].value_counts())
        return df
    elif code_type =='FrequencyEncoding':
        #频率编码
        df[fea] = df[fea].map(df[fea].value_counts())/len(df)
        return df
    
def data_bins(train,code_fea,code_type):
    '''
    目标：编码
    train：目标数据集
    code_fea：需编码的字段
    code_type：编码类型
    '''
    for i in range(0,len(feas)):
        fea = code_fea[i]
        code_type_1 = code_type[i]
        train =coding(train,fea,code_type_1)
    return train