##!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pymysql
from sqlalchemy import create_engine
pymysql.install_as_MySQLdb()
from urllib.parse import quote_plus as urlquote
import requests
import json


##############训练所需配置读取及准备#############
def datadb_conn_param():
    '''
    配置表数据库信息及连接
    '''
    user = 'ddframework_dev'
    passwd = 'Ddframework_dev@20211201'
    host = '192.168.22.72'
    port = 3310
    db = 'dev_ddframework'
    DB_CONNECT = f'mysql+pymysql://{user}:{urlquote(passwd)}@{host}:{port}/{db}?charset=utf8'
    engine = create_engine(DB_CONNECT)
    #库连接
    conn = pymysql.connect( #创建数据库连接
                           host = host, 
                           port = port, 
                           user = user, 
                           passwd = passwd,
                           db = db,
                           charset="utf8"  
    )
    cursor = conn.cursor()
    return engine,cursor,conn

def read_param(train_id):
    '''
    配置读取及准备
    train_id：训练id
    '''
    engine,cursor,conn_1 = datadb_conn_param()
    #训练id
    param_0 = pd.read_sql('SELECT id,model_info_id FROM ddf_model_training_history', con=conn_1)
    #配置
    param_1 = pd.read_sql('SELECT id,model_type FROM ddf_model_info', con=conn_1)
    param_2 = pd.read_sql('SELECT model_info_id,y_value_field_name,table_name,data_source_id,database_name FROM ddf_model_datasource', con=conn_1)
    param_3 = pd.read_sql('SELECT model_info_id,app_config_id,param_value,field_name FROM ddf_model_data_processing', con=conn_1)
    param_4 = pd.read_sql('SELECT model_info_id,app_config_id,param_value,field_name FROM ddf_model_feature_engineering', con=conn_1)
    param_5 = pd.read_sql('SELECT model_info_id,training_set_percent FROM ddf_model_data_partitioning', con=conn_1)
    param_6 = pd.read_sql('SELECT model_info_id,app_config_id FROM ddf_model_algorithm', con=conn_1)
#     param_7 = pd.read_sql('SELECT * FROM ddf_application', con=conn_1)##编码转换表
    param_8 = pd.read_sql('SELECT datasource_config_id,db_ip,db_username,db_port FROM ddf_datasource_db_collection_config', con=conn_1)#数据库信息表
    url = 'https://192.168.22.71:9002/ddfBootApi' ##文件上传、下载、数据库密码的url
    #模型id
    model_id = param_0[param_0['id']==train_id]['model_info_id'].values[0]
    #基本信息
    model_type = param_1[param_1['id']==model_id]['model_type'].values[0]
    #数据选取
    label = param_2[param_2['model_info_id']==model_id]['y_value_field_name'].values[0]
    table_name = param_2[param_2['model_info_id']==model_id]['table_name'].values[0]
    data_source_id = param_2[param_2['model_info_id']==model_id]['data_source_id'].values[0]
    db = param_2[param_2['model_info_id']==model_id]['database_name'].values[0]
    host =param_8[param_8['datasource_config_id']==data_source_id]['db_ip'].values[0]
    port =int(param_8[param_8['datasource_config_id']==data_source_id]['db_port'].values[0])
    user =param_8[param_8['datasource_config_id']==data_source_id]['db_username'].values[0]
    passwd = sql_passwd(url,data_source_id)#数据集数据库密码以接口形式传递
#     print(passwd)
    #数据处理类别
    data_processing = set(param_3[param_3['model_info_id']==model_id]['app_config_id'])
    #缺失率参数
    miss_rate = param_3[(param_3['model_info_id']==model_id)&\
            (param_3['app_config_id']==7)].drop_duplicates(subset=['app_config_id'])#
    if miss_rate.shape[0]!=0:
        miss_rate = float(miss_rate['param_value'].values[0])
    #编码参数
    #编码
    code_fea = param_3[(param_3['model_info_id']==model_id)&(param_4['app_config_id']==8)]['field_name'].tolist()
    code_type = param_3[(param_3['model_info_id']==model_id)&(param_4['app_config_id']==8)]['param_value'].tolist()
    #特征工程类别
    engineering = set(param_4[param_4['model_info_id']==model_id]['app_config_id'])
    #相关系数参数
    corr_y = param_4[(param_4['model_info_id']==model_id)&\
            (param_4['app_config_id']==9)].drop_duplicates(subset=['app_config_id'])#
    if corr_y.shape[0]!=0:
        corr_y = float(corr_y['param_value'].values[0])
    #分箱的参数
    bin_feas = param_4[(param_4['model_info_id']==model_id)&(param_4['app_config_id']==10)]['field_name'].tolist()
    bins = param_4[(param_4['model_info_id']==model_id)&(param_4['app_config_id']==10)]['param_value'].tolist()
    #数据划分
    split_rate = param_5[param_5['model_info_id']==model_id]['training_set_percent'].values[0]/100
    #算法选择
    algorithm = param_6[param_6['model_info_id']==model_id]['app_config_id'].values[0]
    title_0 = '配置读取'
    tip_0 = '完成模型配置读取及获取'
    cursor.close()
    conn_1.close()
    return model_id,model_type,label,table_name,db,host,port,user,passwd,data_processing,miss_rate,code_fea,code_type,engineering,corr_y,bin_feas,bins,split_rate,algorithm,url,title_0,tip_0

#数据集的数据库密码获取
def sql_passwd(url,data_source_id):
    ''''
    数据集所在数据库密码获取
    url：数据上传下载接口url
    data_source_id：参数
    '''
#     url = 'https://192.168.22.71:9002/ddfBootApi'
    url_sql = url+'/dbCollectionConfig/web/databasePasswordEncryptEcbByDatasourceConfigId'
    payload='datasourceConfigId=%s'%(data_source_id)
    headers = {
      'Content-Type': 'application/x-www-form-urlencoded'
    }
    response = requests.request("POST", url_sql, headers=headers, data=payload,verify=False)
    response = response.text
    response = json.loads(response)
    mima = response['data']
    if mima=='':
        print('接口错误，返回数据库密码为空')
    return mima

#数据集数据库信息
def datadb_conn_data(train_id):
    '''
    数据集所在数据库信息及连接
    创建游标将数据写回相应数据库
    train_id：训练id
    '''
    model_id,model_type,label,table_name,db,host,port,user,passwd,data_processing,miss_rate,code_fea,code_type,engineering,corr_y,bin_feas,bins,split_rate,algorithm,url,title_0,tip_0 = read_param(train_id)
    DB_CONNECT = f'mysql+pymysql://{user}:{urlquote(passwd)}@{host}:{port}/{db}?charset=utf8'
    engine = create_engine(DB_CONNECT)
    #库连接
    conn = pymysql.connect( #创建数据库连接
                           host = host, 
                           port = port, 
                           user = user, 
                           passwd = passwd,
                           db = db,
                           charset="utf8"  
    )
    cursor = conn.cursor()
    return engine,cursor,conn

def log_return(model_id, train_id, title,tip):
    '''
    目的：日志输出
    model_id：模型id
    train_id：训练id
    title：标题
    tip：具体内容
    '''
    engine,cursor,conn = datadb_conn_param()
    cursor.execute('insert into ddf_model_training_log(model_info_id, training_id, title,content) values(%s, %s, "%s","%s")' % (model_id, train_id, title,tip))
    cursor.execute('commit')
    return tip