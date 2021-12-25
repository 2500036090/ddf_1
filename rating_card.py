##!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np


################输出评分卡###############
def woe_rule(feature_cols,df_woe):
    '''
    目的：得到WOE规则
    feature_cols：特征集
    df_woe：完成woe转换的数据集
    '''
    df_bin_to_woe = pd.DataFrame(columns=['feature', 'bin', 'woe'])
    for feature in feature_cols:
        b = 'bin_' + feature
        w = 'woe_bin_' + feature
        # 得到该feature的规则
        df = df_woe[[w, b]].drop_duplicates()
        df.columns = ['woe', 'bin']
        df['feature'] = feature
        # 拼接到大的规则表
        df_bin_to_woe = pd.concat([df_bin_to_woe, df])
    return df_bin_to_woe

def generate_scorecard(model_coef,df_woe, features):
    '''
    目的：得到评分卡
    model_coef：模型参数
    df_woe：完成woe转换的数据集
    features：特征集
    '''
    A = 1000 # 基础分
    B = 42.13 # 刻度 50/math.log(2)
    df_rule = woe_rule(features,df_woe)
    lst = []
    cols = ['Variable', 'Binning', 'Score']
    # 模型系数
    coef = model_coef[0]
    for i in range(len(features)):
        feature = features[i]
        # 筛选feature的WOE规则
        df = df_rule[df_rule['feature'] == feature]
        for index, row in df.iterrows():
            # Variable, Binning, Score
            lst.append([feature, row['bin'], int(-row['woe']*coef[i]*B)])
    data = pd.DataFrame(lst, columns=cols)
    return data

#########设计自动评分#########
def str_to_int(s):
    if s == 'inf':
        return 999999
    if s == '-inf':
        return -999999
    return float(s)

def map_value_to_bin(feature_value, feature_to_bin):
    '''
    目的：将value映射到bin
    feature_value：特征值
    feature_to_bin：特征值对应的箱
    '''
    for index, row in feature_to_bin.iterrows():
        bins = str(row['Binning'])
        # 是否为左开
        left_open = bins[0] == '('
        # 是否为右开
        right_open = bins[-1] == ')'
        binnings = bins[1:-1].split(',')
        
        in_range = True # 假设是在区间范围内
        # 判断是否为左开
        if left_open:
            if feature_value<=str_to_int(binnings[0]):
                in_range = False
            else: #[
                if feature_value < str_to_int(binnings[0]):
                    in_range = False
        # 判断是否为右开
        if right_open:
            if feature_value >= str_to_int(binnings[1]):
                in_range = False
            else:
                if feature_value > str_to_int(binnings[1]):
                    in_range = False
        if in_range:
            return row['Binning']
    return 'null'
    
def map_to_score(df, score_card):
    '''
    目的：根据score_card，对df计算分数
    df：数据集
    score_card：评分卡规则
    '''
    # 得到评分卡规则中的字段
    scored_columns = score_card['Variable'].unique()
    score = 0
    for col in scored_columns:
        # 得到关于col的规则
        feature_to_bin = score_card[score_card['Variable']==col]
        # 计算df中的col的取值
        feature_value = df[col]
        # 讲col的数值 映射到Binning
        selected_bin = map_value_to_bin(feature_value, feature_to_bin)
        # 累加score
        selected_record = feature_to_bin[feature_to_bin['Binning'] == selected_bin]['Score'].iloc[0]
        score += selected_record
    return score

#map_to_score(df_train, score_card)

def calcuate_score_with_card(df, score_card):
    '''
    目的：计算总分
    df：数据集
    score_card：评分卡
    '''
    A = 1000
    # 计算score
    df['score'] = df.apply(map_to_score, args=(score_card, ), axis=1)
    df['score'] += A
    return df