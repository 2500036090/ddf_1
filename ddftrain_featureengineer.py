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


############特征工程############
def corr_of_label(df,label,corr_y):
    '''
    目标：相关性指标筛选
    df:数据集
    label:目标变量
    corr_y:相关性阈值，如0.1
    '''
    #计算相关性
    corrmat = df.corr()
    top_corr_features = corrmat.index[abs(corrmat[label])>corr_y]
    #print(top_corr_features)
    df = df[top_corr_features]
    return df

def data_bins(train,feas,bins):
    '''
    目标：分箱
    train：目标数据集
    feas：需分箱的字段
    bins：需分箱的区间
    '''
    fea_bins = train.columns
    for i in range(0,len(feas)):
        bin_0 = bins[i].replace("[",'').replace("]",'').replace("\"",'')
        bin_0 = list(map(float, bin_0.split(',')))
        fea = feas[i]
        if fea in fea_bins:
            train['bin_'+fea] =pd.cut(train[fea],bin_0)
    return train

def cal_WOE(df, features, target):
    '''
    目的：计算字段的WOE特征值
    df：数据集
    features：需做转换特征
    target：目标字段
    '''
    for feature in features:
        # sum 坏样本个数, count 一共有多少个
        df_woe = df.groupby(feature).agg({target: ['sum', 'count']})
        df_woe.columns = list(map(''.join, df_woe.columns.values))
        df_woe = df_woe.reset_index()
        df_woe.rename(columns={target+'sum': 'bad', target+'count': 'all'}, inplace=True)
        df_woe['good'] = df_woe['all'] - df_woe['bad']
        df_woe['Margin Bad'] = df_woe['bad'] / df_woe['bad'].sum()
        df_woe['Margin Good'] = df_woe['good'] / df_woe['good'].sum()
        # 为了避免分母为0，使用np.log1p
        df_woe['woe'] = np.log1p(df_woe['Margin Bad'] / df_woe['Margin Good'])
        df_woe.columns = [c if c==feature else c+'_'+feature for c in df_woe.columns]
        # 拼接
        df = df.merge(df_woe, on=feature, how='left')
    return df

def woe_tran(train,label):
    '''
    目的：特征woe转换
    train：训练集
    label：目标特征
    '''
    # 统计分箱的字段
    bin_cols = [x for x in train.columns if x.startswith('bin_')]
    df_woe = cal_WOE(train, bin_cols, label)
    feature_cols = list(train.columns)
    feature_cols.remove('label')
    for i in bin_cols:
        feature_cols.remove(i)
    woe_cols = ['woe_bin_' + c for c in feature_cols]+[label]
#     for col in woe_cols:
#         print(col, df_woe[col].unique())
    train = df_woe[woe_cols] #woe_cols,
    return train,df_woe,feature_cols