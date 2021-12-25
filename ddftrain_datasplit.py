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


############数据集划分############
def split_data(train,label,split_rate):
    '''
    data:训练集（只包含训练所需字段及label）
    label：预测目标，如'label'
    split_rate:划分比例，如0.8
    '''    
    fea_all = train.columns.tolist()
    fea_out = [label,'Index','masterdataid','MASTERDATAID','id','enrol','creditlevel']
    n = list(set(fea_all))
    n.sort(key = fea_all.index)
    m = list(set(fea_out))
    for i in m:
        if i in n:
            n.remove(i)
    train_fea = n
    X = train[train_fea].reset_index(drop=True)  #(5039, 14)
    y = train[label].reset_index(drop=True)
    X_train, X_test, y_train, y_test = train_test_split(X,  y, train_size=split_rate, random_state=2021)
    title_4 = '数据划分'
    tip_4 = '完成数据划分，划分比例为训练集为%s。训练集为%s，测试集为%s'%(split_rate,X_train.shape,X_test.shape)
    return X_train, X_test, y_train, y_test,title_4,tip_4