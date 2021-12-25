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
from urllib.parse import quote_plus as urlquote
import time
import datetime
from numpy import inf
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance,plot_tree
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import sys
from rating_card import woe_rule,generate_scorecard
from ddftrain_valplot import ana_ks,auc_confusion,remove_prefix,importance_fea,card_rule_reslut,result_visualization
from ddftrain_getparam import datadb_conn_param,read_param,sql_passwd,datadb_conn_data,log_return
from ddftrain_readdata import read_tabel
from ddftrain_dataprosessing import info,drop_na_unique,coding,data_bins
from ddftrain_featureengineer import corr_of_label,data_bins,cal_WOE,woe_tran
from ddftrain_datasplit import split_data
from ddftrain_modelchoose import model_choose



############主运行函数##############
def main(train_id):
    engine,cursor,conn = datadb_conn_param()#########注意，到时候要替换成datadb_conn函数
    try:
        #训练模型状态更新
        cursor.execute('update ddf_model_training_history set result_flag=%s where id=%s', [1,int(train_id)])
        cursor.execute('commit')
        tip = '程序开始，完成配置数据库连接,'
        title = '配置库连接'
        cursor.execute('insert into ddf_model_training_log(training_id,title, content) values(%s,"%s", "%s")' % (train_id,title,tip))
        cursor.execute('commit')
        print('完成配置数据库连接')
    except Exception as e:
        tip = '配置数据库连接失败，报错为:%s'%e
        title = '配置库连接'
        cursor.execute('insert into ddf_model_training_log(training_id,title, content) values(%s,"%s", "%s")' % (train_id,title,tip))
        cursor.execute('commit')
        print(tip)
        print(e)
    try:
        model_id,model_type,label,table_name,db,host,port,user,passwd,data_processing,miss_rate,code_fea,code_type,engineering,corr_y,bin_feas,bins,split_rate,algorithm,url,title_0,tip_0 = read_param(train_id)
        #配置读取日志
        tip = log_return(model_id, train_id, title_0,tip_0)
        print('完成配置读取')
    except Exception as e:
        tip_0 = '配置读取失败，报错为:%s'%e
        title_0 = '配置读取'
        model_id = 1
        tip = log_return(model_id, train_id, title,tip)
        print(tip_0)
        print(e)
    try:
        #读取数据
        train,title_1,tip_1 = read_tabel(train_id,table_name)
        tip = log_return(model_id, train_id, title_1,tip_1)
        print('完成数据读取')
    except Exception as e:
        tip_1 = '数据获取失败，报错为:%s'%e
        title_1 = '数据集库连接'
        tip = log_return(model_id, train_id, title_1,tip_1)
        print(tip_1)
        print(e)
    try:
        #数据处理
        if 7 in data_processing:
            fea_num_1 = train.shape[1]
            train = drop_na_unique(train,miss_rate)
            fea_num_2 = train.shape[1]
            fea_num = fea_num_1-fea_num_2
            title_2 = '数据处理'
            tip_2 = '完成缺失率清洗，筛出缺失率大于百分之%s的特征，此处筛除了%s个特征。'%(int(miss_rate),fea_num)
            tip = log_return(model_id, train_id, title_2,tip_2)
        if 8 in data_processing:
            train = data_bins(train,code_fea,code_type)
            title_2 = '数据处理'
            tip_2 = '完成对%s特征编码，编码方式为%s。'%(code_fea,code_fea)
            tip = log_return(model_id, train_id, title_2,tip_2)
        print('完成数据处理')
    except Exception as e:
        tip_2 = '数据处理失败，报错为:%s'%e
        title_2 = '数据处理'
        tip = log_return(model_id, train_id, title_2,tip_2)
        print(tip_2)
        print(e)
    try:   
        #特征工程
        if 9 in engineering:
            fea_num_1 = train.shape[1]
            train = corr_of_label(train,label,corr_y)
            fea_num_2 = train.shape[1]
            fea_num = fea_num_1-fea_num_2
            title_3 = '特征工程'
            tip_3 = '完成特征相关性筛选，只选择了相关性高于%s的特征，此次筛选掉了%s个特征'%(corr_y,fea_num)
            tip = log_return(model_id, train_id, title_3,tip_3)
        if 10 in engineering:
            train = data_bins(train,bin_feas,bins)
            title_3 = '特征工程'
            tip_3 = '完成特征分箱，此次完成了对%s这些字段的分箱'%(bin_feas)
            tip = log_return(model_id, train_id, title_3,tip_3)
        if 11 in engineering:
            train,df_woe,feature_cols = woe_tran(train,label)
            fea_num= train.shape[1]-1
            title_3 = '特征工程'
            tip_3 = '完成特征woe转换，此次完成了%s个特征的woe转换'%(fea_num)
            tip = log_return(model_id, train_id, title_3,tip_3)
        print('完成特征工程')
    except Exception as e:
        tip_3 = '特征工程失败，报错为:%s'%e
        title_3 = '特征工程'
        tip = log_return(model_id, train_id, title_3,tip_3)
        print(tip_3)
        print(e)
    try:   
        #数据划分
        X_train, X_test, y_train, y_test,title_4,tip_4 = split_data(train,label,split_rate)
        tip = log_return(model_id, train_id, title_4,tip_4)
        print('完成数据划分')
    except Exception as e:
        tip_4 = '数据划分失败，报错为:%s'%e
        title_4 = '数据划分'
        tip = log_return(model_id, train_id, title_4,tip_4)
        print(tip_4)
        print(e)
    try:   
        #训练模型
        model,title_5 ,tip_5,ts = model_choose(X_train,y_train,X_test,algorithm,model_type,train_id,url)
        tip = log_return(model_id, train_id, title_5,tip_5)
        print(tip_5)
    except Exception as e:
        tip_5 = '模型训练失败，报错为:%s'%e
        title_5 = '模型训练'
        tip = log_return(model_id, train_id, title_5,tip_5)
        print(tip_5)
        print(e)
    try:  
        #验证结果输出
        if model_type==1:
            card_rule_reslut(model, df_woe, feature_cols,model_id,train_id)
        title_6,tip_6 = result_visualization(model,X_train,X_test,y_test,algorithm,model_id,train_id,model_type)
        tip = log_return(model_id, train_id, title_6,tip_6)
        print('完成验证结果输出，程序结束')
    except Exception as e:
        tip_6 = '模型验证结果输出失败，程序中断，报错为%s'%e
        title_6 = '模型验证'
        tip = log_return(model_id, train_id, title_6,tip_6)
        print(tip_6)
        print(e)
    cursor.close()
    conn.close()
    return ('程序结束')


    

if __name__ == '__main__':
    train_id = int(sys.argv[1])
    main(train_id)
