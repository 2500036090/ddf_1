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
import joblib
import requests
import json
from ddftrain_getparam import datadb_conn_param,read_param,sql_passwd,datadb_conn_data,log_return
import os


################模型训练##############
def model_upload(url,train_id):
    '''
    目的：上传模型文件
    url:上传文件的接口
    train_id：训练id
    '''
    url_upload = url+'/cmsfile/web/addCmsFile'
    payload={}
    files=[('file',('ddf_model.m',open('ddf_model.m','rb'),'application/octet-stream'))
    ]
    headers = {}
    response = requests.request("POST", url_upload, headers=headers, data=payload, files=files,verify=False)
    data = json.loads(response.text)
    try:
        id_load = int(data['data']['id'])
        print('返回上传文件id为%s'%id_load)
    except Exception as e:
        print('接口错误，返回上传文件id为空，报错为%s'%e)
    engine,cursor,conn = datadb_conn_param()
    #保存上传id
    cursor.execute('update ddf_model_training_history set algorithm_file_id=%s where id=%s', [id_load,int(train_id)])
    conn.commit()
    ts = '完成模型文件上传,上传文件返回密码为：%s'%id_load
    return ts

def model_choose(X_train,y_train,X_test,algorithm,model_type,train_id,url):
    '''
    目的：算法选择
    X_train,y_train,X_test：训练集及验证集
    algorithm,model_type：算法配置、模型类型
    train_id,url：训练id、上传文件url
    '''
    if algorithm == 12:
        #逻辑回归
        model = LogisticRegression()
        algorithm = '逻辑回归算法'
    elif algorithm == 13:
        #决策树分类
        model = tree.DecisionTreeClassifier(criterion='gini',max_depth=7, min_weight_fraction_leaf=0.01,random_state=30,)
        algorithm = '决策树算法'
    elif algorithm == 14:
        #xgb分类
        model = xgb.XGBClassifier(max_depth=10, n_estimators=10)#, objective='multi:softmax', silent=True
        algorithm = 'XGB算法'
    else:
        model = algorithm
        title_5 = '模型训练'
        tip_5 = '模型训练失败，所选模型不存在'
        ts = '上传文件失败'
        return model,title_5 ,tip_5,ts
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ## 保存模型
    joblib.dump(model,'ddf_model.m',compress=9)
    #上传模型
    ts = model_upload(url,train_id)
    if os.path.exists('ddf_model.m'):
        os.remove('ddf_model.m')
    else:
        print("没有该文件")
    title_5 = '模型训练'
    if model_type==1:
        tip_5 = '完成模型训练，使用%s完成评分卡模型,并完成模型文件上传'%(algorithm)
    else:
        tip_5 = '完成模型训练，使用%s完成分类预测模型,并完成模型文件上传'%(algorithm)
    return model,title_5 ,tip_5,ts
    
