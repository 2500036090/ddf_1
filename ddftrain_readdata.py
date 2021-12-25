##!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import pymysql
from sqlalchemy import create_engine
pymysql.install_as_MySQLdb()
from ddftrain_getparam import datadb_conn_param,read_param,sql_passwd,datadb_conn_data,log_return



###############读取数据集########
def read_tabel(train_id,table):
    '''
    数据集读取
    train_id：训练id
    table：表名
    '''
    #库连接
    engine,cursor,conn = datadb_conn_data(train_id)
    #读取数据
    sql='SELECT * FROM %s'%table
    train = pd.read_sql(sql, con=conn)
    title_1 = '数据读取'
    tip_1 = '完成数据读取，包含了%s个字段,%s行样本'% (train.shape[1],train.shape[0])
    return train,title_1,tip_1